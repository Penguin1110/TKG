"use strict";

const $ = (id) => document.getElementById(id);
const COLORS = {
  edge: "#35505a", edgeHover: "#789ba6", node: "#42b8c8", stub: "#72838b",
  trajectoryOnly: "#9c769d", temporal: "#b083d1", path: "#e6a43a", current: "#ed6b4f", target: "#4d9d76",
  label: "#d9e7ea", ink: "#10212b"
};

const state = {
  data: null, nodes: [], edges: [], nodeById: new Map(), edgeById: new Map(), adjacency: new Map(),
  visibleNodes: [], visibleIds: new Set(), visibleEdgeIds: new Set(), nodeRevealAt: new Map(), trajectory: null, cursor: 0,
  playing: false, eventStart: 0, eventProgress: 0, speed: 1,
  hover: null, selected: null, draggingNode: null, panning: false,
  pointerStart: null, viewStart: null, layoutTicks: 0, layoutRunning: false,
  view: {x: 0, y: 0, scale: 1}, dpr: 1
};

const canvas = $("graph-canvas");
const ctx = canvas.getContext("2d");

function hashCode(value) {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i++) { hash ^= value.charCodeAt(i); hash = Math.imul(hash, 16777619); }
  return hash >>> 0;
}

function initializePositions(nodes) {
  const groups = [...new Set(nodes.map(n => n.snapshot_group))].sort();
  const groupIndex = new Map(groups.map((g, i) => [g, i]));
  const columns = Math.ceil(Math.sqrt(groups.length));
  nodes.forEach((node, index) => {
    const h = hashCode(node.id);
    const g = groupIndex.get(node.snapshot_group) || 0;
    const gx = (g % columns) * 680;
    const gy = Math.floor(g / columns) * 520;
    const angle = (h % 6283) / 1000;
    const radius = 60 + ((h >>> 8) % 320);
    node.x = gx + Math.cos(angle) * radius;
    node.y = gy + Math.sin(angle) * radius;
    node.vx = 0; node.vy = 0; node.radius = 5 + Math.min(9, Math.sqrt((node.in_degree || 0) + (node.out_degree || 0)) * 1.35);
    node._index = index;
  });
  state.layoutTicks = 0; state.layoutRunning = true;
  $("layout-status").textContent = "LAYOUT SETTLING";
}

function buildIndexes() {
  state.nodeById = new Map(state.nodes.map(n => [n.id, n]));
  state.edgeById = new Map(state.edges.map(edge => [edge.id, edge]));
  state.adjacency = new Map(state.nodes.map(n => [n.id, new Set()]));
  state.edges.forEach(edge => {
    state.adjacency.get(edge.source)?.add(edge.target);
    state.adjacency.get(edge.target)?.add(edge.source);
  });
}

function populateSelect(id, values, allLabel) {
  const select = $(id); const prior = select.value;
  select.replaceChildren(new Option(allLabel, ""), ...values.map(v => new Option(v, v)));
  if (values.includes(prior)) select.value = prior;
}

function trajectoryMatches(t) {
  return (!$("model-filter").value || t.model === $("model-filter").value) &&
    (!$("case-filter").value || t.case_id === $("case-filter").value) &&
    (!$("arm-filter").value || t.arm === $("arm-filter").value);
}

function positionNewRevealNodes(newIds) {
  if (!state.trajectory || !newIds.length) return;
  const unplaced = new Set(newIds);
  const pivot = state.nodeById.get(state.trajectory.pivot_node_id);
  if (pivot && unplaced.has(pivot.id)) {
    pivot.x = 0; pivot.y = 0; pivot.vx = 0; pivot.vy = 0; unplaced.delete(pivot.id);
  }
  state.trajectory.events.slice(0, state.cursor).forEach(event => {
    const center = state.nodeById.get(event.expansion_node_id);
    if (!center) return;
    if (unplaced.has(center.id)) {
      let anchor = event.action === "enter_start_page"
        ? pivot : state.nodeById.get(event.from_node_id);
      if (!anchor || anchor.id === center.id) anchor = pivot;
      if (anchor && anchor.id !== center.id) {
        const angle = (hashCode(center.id) % 6283) / 1000;
        center.x = anchor.x + Math.cos(angle) * 220;
        center.y = anchor.y + Math.sin(angle) * 220;
      }
      center.vx = 0; center.vy = 0; unplaced.delete(center.id);
    }
    (event.reveal_node_ids || []).filter(id => id !== center.id).forEach((id, index) => {
      if (!unplaced.has(id)) return;
      const node = state.nodeById.get(id); if (!node) return;
      const angle = ((hashCode(`${center.id}|${id}`) % 6283) / 1000) + index * .31;
      const radius = 90 + (index % 4) * 24;
      node.x = center.x + Math.cos(angle) * radius;
      node.y = center.y + Math.sin(angle) * radius;
      node.vx = 0; node.vy = 0; unplaced.delete(id);
    });
  });
}

function refreshTrajectoryOptions() {
  const select = $("trajectory-select"); const prior = select.value;
  const options = state.data.trajectories.filter(trajectoryMatches);
  select.replaceChildren(new Option(options.length ? "Select a run…" : "No matching runs", ""));
  options.forEach(t => {
    const flags = `${t.page_hit ? "hit" : "miss"}/${t.eligible ? "eligible" : "blocked"}`;
    const snapshot = t.arm === "temporal"
      ? "multi-time" : (t.snapshot_selection?.selection_token || "prebound");
    const label = `${t.model} · ${t.case_id} · ${t.arm} · ${snapshot} · d${t.start_distance ?? "?"} · r${t.repeat ?? 0} · ${flags}`;
    select.add(new Option(label, t.id));
  });
  if ([...select.options].some(o => o.value === prior)) select.value = prior;
  else selectTrajectory("");
}

function refreshVisibility() {
  const snapshot = $("snapshot-filter").value;
  const showStubs = $("show-stubs").checked;
  const filterAllowed = new Set(state.nodes.filter(node =>
    (!snapshot || node.snapshot_group === snapshot) && (showStubs || node.cached)
  ).map(node => node.id));
  const revealIds = new Set();
  const revealEdgeIds = new Set();
  if (state.trajectory) {
    (state.trajectory.initial_reveal_node_ids || []).forEach(id => revealIds.add(id));
    state.trajectory.events.slice(0, state.cursor).forEach(event => {
      (event.reveal_node_ids || []).forEach(id => revealIds.add(id));
      (event.reveal_edge_ids || []).forEach(id => revealEdgeIds.add(id));
    });
  } else {
    state.data.trajectories.forEach(t => { if (t.pivot_node_id) revealIds.add(t.pivot_node_id); });
  }
  const allowed = new Set([...revealIds].filter(id => filterAllowed.has(id)));
  const previous = state.visibleIds;
  const now = performance.now();
  const newlyVisible = [...allowed].filter(id => !previous.has(id));
  positionNewRevealNodes(newlyVisible);
  newlyVisible.forEach(id => state.nodeRevealAt.set(id, now));
  [...previous].filter(id => !allowed.has(id)).forEach(id => state.nodeRevealAt.delete(id));
  if (newlyVisible.length) {
    state.layoutTicks = 0; state.layoutRunning = true;
    $("layout-status").textContent = "LAYOUT SETTLING";
  }
  state.visibleIds = allowed;
  state.visibleEdgeIds = new Set([...revealEdgeIds].filter(id => {
    const edge = state.edgeById.get(id);
    return edge && allowed.has(edge.source) && allowed.has(edge.target);
  }));
  state.visibleNodes = state.nodes.filter(node => allowed.has(node.id));
  $("empty-state").hidden = state.visibleNodes.length > 0;
  $("node-count").textContent = state.visibleNodes.length.toLocaleString();
  $("edge-count").textContent = state.visibleEdgeIds.size.toLocaleString();
}

function selectTrajectory(id) {
  state.playing = false; state.cursor = 0; state.eventProgress = 0;
  state.trajectory = state.data.trajectories.find(t => t.id === id) || null;
  if (state.trajectory?.pivot_node_id) {
    const pivot = state.nodeById.get(state.trajectory.pivot_node_id);
    if (pivot) { pivot.x = 0; pivot.y = 0; pivot.vx = 0; pivot.vy = 0; }
  }
  $("play-button").textContent = "▶";
  const events = state.trajectory?.events || [];
  $("event-slider").disabled = !events.length;
  $("event-slider").max = events.length;
  $("event-slider").value = 0;
  const badges = $("run-badges"); badges.replaceChildren();
  if (state.trajectory) {
    const values = state.trajectory.arm === "temporal" ? [
      [state.trajectory.completed ? "complete" : "partial", state.trajectory.completed],
      [state.trajectory.pk_admitted === true
        ? `PK admitted · old ${state.trajectory.pk_stick_old_count ?? 0}/${state.trajectory.pk_probe_n ?? 0}`
        : "PK admission missing", state.trajectory.pk_admitted === true],
      [state.trajectory.eligible ? "judged" : "judge missing", state.trajectory.eligible],
      [state.trajectory.reasoning_hop_count
        ? `${state.trajectory.reasoning_hop_count}-hop relative`
        : "direct question", !!state.trajectory.reasoning_hop_count],
      [state.trajectory.target_title_revealed === false ? "pivot hidden" : "pivot revealed",
        state.trajectory.target_title_revealed === false],
      [state.trajectory.page_hit ? "pivot hit" : "pivot miss", state.trajectory.page_hit],
      [state.trajectory.shortest_arrival ? "shortest path" : `detour ${state.trajectory.detour_steps ?? "—"}`, state.trajectory.shortest_arrival],
      [state.trajectory.cycle_detected ? `cycle/revisit ${state.trajectory.revisit_count}` : "no revisit", !state.trajectory.cycle_detected],
      [state.trajectory.outcome_stage || "temporal answer", false],
      [state.trajectory.outcome_reason || "unclassified", state.trajectory.outcome_reason === "correct_after"]
    ] : [
      [state.trajectory.completed ? "complete" : "partial", state.trajectory.completed],
      [state.trajectory.page_hit ? "page hit" : "miss", state.trajectory.page_hit],
      [state.trajectory.eligible ? "eligible" : "gate blocked", state.trajectory.eligible],
      [state.trajectory.outcome_stage || "unclassified stage", false],
      [state.trajectory.outcome_reason || "unclassified", state.trajectory.outcome_reason === "no_reversion_observed"]
    ];
    values.forEach(([label, ok]) => { const el = document.createElement("span"); el.className = `badge ${ok ? "ok" : "warn"}`; el.textContent = label; badges.append(el); });
    if (state.trajectory.arm === "temporal") {
      $("snapshot-filter").value = "";
    } else {
      const selectedSnapshot = state.trajectory.snapshot_selection
        ? state.trajectory.snapshot_selection.selected_as_of
        : state.trajectory.snapshot_as_of;
      const snapshotValue = selectedSnapshot || "__CURRENT_SNAPSHOT__";
      if ([...$("snapshot-filter").options].some(option => option.value === snapshotValue)) {
        $("snapshot-filter").value = snapshotValue;
      }
    }
  }
  refreshVisibility(); updateTimeline();
}

function currentNodeId() {
  if (!state.trajectory) return null;
  if (state.cursor <= 0) return state.trajectory.pivot_node_id || null;
  const event = state.trajectory.events[Math.min(state.cursor - 1, state.trajectory.events.length - 1)];
  return event?.to_node_id || state.trajectory.path_node_ids[0] || null;
}

function completedPath() {
  const nodes = new Set(); const edges = new Set();
  if (!state.trajectory) return {nodes, edges};
  const pivot = state.trajectory.pivot_node_id; if (pivot) nodes.add(pivot);
  state.trajectory.events.slice(0, state.cursor).forEach(event => {
    nodes.add(event.from_node_id); nodes.add(event.to_node_id);
    if (event.moved) edges.add(`${event.from_node_id}\u0000${event.to_node_id}`);
  });
  return {nodes, edges};
}

function updateTimeline() {
  refreshVisibility();
  const t = state.trajectory; const total = t?.events.length || 0;
  $("event-counter").textContent = `${state.cursor} / ${total}`;
  $("event-slider").value = state.cursor;
  $("prev-button").disabled = !t || state.cursor <= 0;
  $("next-button").disabled = !t || state.cursor >= total;
  $("play-button").disabled = !t || !total;
  if (!t) {
    $("event-title").textContent = "Choose a trajectory to replay";
    $("event-result").textContent = "The graph remains fully interactive while a run is playing.";
    return;
  }
  if (state.cursor === 0) {
    $("event-title").textContent = `${t.pk_admitted === true ? "PK admitted" : "PK unavailable"} · Start · ${t.start_title || "unknown page"}`;
    const pk = t.pk_admitted === true
      ? `fresh-context PK: new ${t.pk_stick_new_count ?? 0}/${t.pk_probe_n ?? 0}, old ${t.pk_stick_old_count ?? 0}/${t.pk_probe_n ?? 0}`
      : `PK: ${t.pk_gate_reason || "missing"}`;
    const cutoff = t.knowledge_cutoff?.cutoff_date
      ? ` · cutoff ${t.knowledge_cutoff.cutoff_date}` : "";
    const hops = t.reasoning_hop_count ? ` · ${t.reasoning_hop_count} reasoning hops` : "";
    $("event-result").textContent = `${pk}${cutoff}${hops} · ${t.model} · ${t.outcome_stage || "unclassified"} / ${t.outcome_reason || t.stop_reason || "unknown"}`;
    return;
  }
  const event = t.events[state.cursor - 1];
  $("event-title").textContent = `Step ${event.step ?? state.cursor} · ${event.action} · ${event.from_title}${event.moved ? ` → ${event.to_title}` : ""}`;
  const progress = event.navigation_step != null
    ? `nav ${event.navigation_step} · d→pivot ${event.distance_to_pivot ?? "outside"}${event.revisited ? " · revisited" : ""} · ` : "";
  $("event-result").textContent = progress + (event.result || event.free_text || "No tool output recorded.");
}

function advance(delta) {
  if (!state.trajectory) return;
  state.playing = false; $("play-button").textContent = "▶";
  state.cursor = Math.max(0, Math.min(state.trajectory.events.length, state.cursor + delta));
  state.eventProgress = 0; updateTimeline(); fitGraph();
}

function togglePlay() {
  if (!state.trajectory?.events.length) return;
  if (state.cursor >= state.trajectory.events.length) state.cursor = 0;
  state.playing = !state.playing; state.eventStart = performance.now(); state.eventProgress = 0;
  $("play-button").textContent = state.playing ? "Ⅱ" : "▶";
  updateTimeline();
}

function updatePlayback(now) {
  if (!state.playing || !state.trajectory) return;
  const event = state.trajectory.events[state.cursor];
  if (!event) { state.playing = false; $("play-button").textContent = "▶"; return; }
  const duration = (event.moved ? 1300 : 620) / state.speed;
  state.eventProgress = Math.min(1, (now - state.eventStart) / duration);
  if (state.eventProgress >= 1) {
    state.cursor += 1; state.eventProgress = 0; state.eventStart = now; updateTimeline(); fitGraph();
    if (state.cursor >= state.trajectory.events.length) { state.playing = false; $("play-button").textContent = "▶"; }
  }
}

function physicsStep() {
  if (!state.layoutRunning) return;
  const nodes = state.visibleNodes;
  const visible = state.visibleIds;
  if (!nodes.length) return;
  const damping = .86;
  state.edges.forEach(edge => {
    if (!visible.has(edge.source) || !visible.has(edge.target)) return;
    const a = state.nodeById.get(edge.source), b = state.nodeById.get(edge.target); if (!a || !b) return;
    const dx = b.x - a.x, dy = b.y - a.y, dist = Math.hypot(dx,dy) || 1;
    const ideal = 90 + Math.min(75, (a.radius + b.radius) * 3); const force = (dist - ideal) * .0018;
    const fx = dx / dist * force, fy = dy / dist * force;
    if (a !== state.draggingNode) { a.vx += fx; a.vy += fy; }
    if (b !== state.draggingNode) { b.vx -= fx; b.vy -= fy; }
  });
  const cell = 85, grid = new Map();
  nodes.forEach(n => { const key = `${Math.floor(n.x/cell)},${Math.floor(n.y/cell)}`; if (!grid.has(key)) grid.set(key,[]); grid.get(key).push(n); });
  nodes.forEach(a => {
    const cx=Math.floor(a.x/cell), cy=Math.floor(a.y/cell);
    for(let gx=cx-1;gx<=cx+1;gx++) for(let gy=cy-1;gy<=cy+1;gy++) (grid.get(`${gx},${gy}`)||[]).forEach(b=>{
      if (a._index >= b._index) return; let dx=b.x-a.x,dy=b.y-a.y,dist=Math.hypot(dx,dy)||.1;
      const minimum=a.radius+b.radius+18; if(dist<minimum){ const f=(minimum-dist)*.012; dx/=dist;dy/=dist; a.vx-=dx*f;a.vy-=dy*f;b.vx+=dx*f;b.vy+=dy*f; }
    });
    a.vx += -a.x * .000025; a.vy += -a.y * .000025;
  });
  let energy=0;
  nodes.forEach(n => { if(n===state.draggingNode)return; n.vx*=damping;n.vy*=damping; n.x+=n.vx;n.y+=n.vy; energy+=Math.abs(n.vx)+Math.abs(n.vy); });
  state.layoutTicks++;
  if (state.layoutTicks > 420 || (state.layoutTicks > 100 && energy/nodes.length < .018)) {
    state.layoutRunning=false; $("layout-status").textContent="LAYOUT READY";
  }
}

function resizeCanvas() {
  const rect=canvas.getBoundingClientRect(); state.dpr=Math.min(2,window.devicePixelRatio||1);
  canvas.width=Math.round(rect.width*state.dpr); canvas.height=Math.round(rect.height*state.dpr);
}
function screenToWorld(x,y){return{x:(x-state.view.x)/state.view.scale,y:(y-state.view.y)/state.view.scale};}
function worldToScreen(x,y){return{x:x*state.view.scale+state.view.x,y:y*state.view.scale+state.view.y};}

function drawArrow(a,b,color,alpha,width) {
  const dx=b.x-a.x,dy=b.y-a.y,dist=Math.hypot(dx,dy); if(!dist)return;
  const ux=dx/dist,uy=dy/dist,start=a.radius+2,end=b.radius+5;
  const x1=a.x+ux*start,y1=a.y+uy*start,x2=b.x-ux*end,y2=b.y-uy*end;
  ctx.globalAlpha=alpha;ctx.strokeStyle=color;ctx.lineWidth=width/state.view.scale;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();
  if(state.view.scale>.42){const size=5/state.view.scale;ctx.fillStyle=color;ctx.beginPath();ctx.moveTo(x2,y2);ctx.lineTo(x2-ux*size-uy*size*.65,y2-uy*size+ux*size*.65);ctx.lineTo(x2-ux*size+uy*size*.65,y2-uy*size-ux*size*.65);ctx.closePath();ctx.fill();}
}

function draw(now) {
  const rect=canvas.getBoundingClientRect(); ctx.setTransform(state.dpr,0,0,state.dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);
  ctx.save();ctx.translate(state.view.x,state.view.y);ctx.scale(state.view.scale,state.view.scale);
  const path=completedPath(), hoverId=state.hover?.id, currentId=currentNodeId();
  state.edges.forEach(edge=>{
    if(!state.visibleEdgeIds.has(edge.id))return;
    const a=state.nodeById.get(edge.source),b=state.nodeById.get(edge.target);if(!a||!b)return;
    const key=`${edge.source}\u0000${edge.target}`; const onPath=path.edges.has(key);
    const nearHover=hoverId&&(edge.source===hoverId||edge.target===hoverId);
    const temporal=edge.kind==="temporal";
    drawArrow(a,b,temporal?COLORS.temporal:(onPath?COLORS.path:(nearHover?COLORS.edgeHover:COLORS.edge)),onPath?.95:(nearHover?.7:.24),onPath?2.8:(temporal?2:1));
  });
  state.visibleNodes.forEach(node=>{
    const active=path.nodes.has(node.id),current=node.id===currentId,selected=node.id===state.selected?.id,hover=node.id===hoverId;
    const isPivot=node.id===state.trajectory?.pivot_node_id;
    const revealAlpha=Math.min(1,Math.max(.08,(now-(state.nodeRevealAt.get(node.id)||0))/420));
    const pulse=current?(1+Math.sin(now/180)*.16):1; const radius=node.radius*pulse;
    if(current){ctx.globalAlpha=.2*revealAlpha;ctx.fillStyle=COLORS.current;ctx.beginPath();ctx.arc(node.x,node.y,radius+10/state.view.scale,0,Math.PI*2);ctx.fill();}
    ctx.globalAlpha=revealAlpha;ctx.fillStyle=current?COLORS.current:(isPivot?COLORS.target:(active?COLORS.path:(node.trajectory_only?COLORS.trajectoryOnly:(node.cached?COLORS.node:COLORS.stub))));
    ctx.beginPath();ctx.arc(node.x,node.y,radius,0,Math.PI*2);ctx.fill();
    ctx.lineWidth=(selected||hover?2.2:1)/state.view.scale;ctx.strokeStyle=selected?"#fff":(hover?COLORS.gold:"#0b1820");ctx.stroke();
    if(state.view.scale>.7||active||hover||selected||isPivot){ctx.globalAlpha=revealAlpha*(active||hover||selected||isPivot?1:.72);ctx.fillStyle=COLORS.label;ctx.font=`${active||isPivot?600:400} ${Math.max(9,11/state.view.scale)}px system-ui`;ctx.textAlign="center";ctx.fillText(node.title,node.x,node.y+radius+13/state.view.scale);}
  });
  if(state.playing&&state.trajectory){const event=state.trajectory.events[state.cursor];if(event){const a=state.nodeById.get(event.from_node_id),b=state.nodeById.get(event.to_node_id);if(a&&b&&state.visibleIds.has(a.id)&&state.visibleIds.has(b.id)){const t=event.moved?state.eventProgress:0;const x=a.x+(b.x-a.x)*t,y=a.y+(b.y-a.y)*t;ctx.globalAlpha=1;ctx.fillStyle="#fff3cf";ctx.beginPath();ctx.arc(x,y,5.5/state.view.scale,0,Math.PI*2);ctx.fill();ctx.strokeStyle=COLORS.current;ctx.lineWidth=2/state.view.scale;ctx.stroke();}}}
  ctx.restore();ctx.globalAlpha=1;
}

function fitGraph(ids=null, animate=false) {
  const nodes=state.nodes.filter(n=>(ids?ids.has(n.id):state.visibleIds.has(n.id))); if(!nodes.length)return;
  const rect=canvas.getBoundingClientRect();let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
  nodes.forEach(n=>{minX=Math.min(minX,n.x);minY=Math.min(minY,n.y);maxX=Math.max(maxX,n.x);maxY=Math.max(maxY,n.y);});
  const width=Math.max(100,maxX-minX),height=Math.max(100,maxY-minY);state.view.scale=Math.max(.08,Math.min(2.4,Math.min((rect.width-120)/width,(rect.height-210)/height)));
  state.view.x=rect.width/2-(minX+maxX)/2*state.view.scale;state.view.y=(rect.height-55)/2-(minY+maxY)/2*state.view.scale;
}

function hitTest(clientX,clientY){const rect=canvas.getBoundingClientRect(),world=screenToWorld(clientX-rect.left,clientY-rect.top);let hit=null,best=Infinity;state.visibleNodes.forEach(n=>{const d=Math.hypot(n.x-world.x,n.y-world.y);if(d<(n.radius+7/state.view.scale)&&d<best){hit=n;best=d;}});return hit;}

function trajectoryDistance(node){return state.trajectory?.distance_by_node_id?.[node.id]??node.distance_to_pivot??null;}
function showTooltip(node,event){if(!node){$("tooltip").hidden=true;return;}$("tooltip-title").textContent=node.title;$("tooltip-meta").textContent=`${node.cached?"cached":"unresolved"} · rev ${node.revision_id??"—"} · ${node.timestamp??node.snapshot_group??"unknown time"} · d→pivot ${trajectoryDistance(node)??"—"} · in ${node.in_degree} / out ${node.out_degree}`;$("tooltip-excerpt").textContent=node.excerpt||"No excerpt stored.";const tip=$("tooltip");tip.hidden=false;let left=event.clientX+16,top=event.clientY+16;if(left+285>innerWidth)left=event.clientX-285;if(top+160>innerHeight)top=event.clientY-165;tip.style.left=`${left}px`;tip.style.top=`${top}px`;}
function showDetails(node){state.selected=node;$("empty-details").hidden=!!node;$("node-details").hidden=!node;if(!node)return;$("detail-title").textContent=node.title;$("detail-revision").textContent=node.revision_id??"not cached";$("detail-timestamp").textContent=node.timestamp??node.snapshot_group;$("detail-degree").textContent=`d→pivot ${trajectoryDistance(node)??"—"} · ${node.in_degree} in / ${node.out_degree} out · ${node.external_link_count||0} external`;$("detail-excerpt").textContent=node.excerpt||"No excerpt stored.";const link=$("detail-link");link.hidden=!node.source_url;if(node.source_url)link.href=node.source_url;}

function focusNode(node){showDetails(node);const rect=canvas.getBoundingClientRect();state.view.scale=Math.max(1.25,state.view.scale);state.view.x=rect.width/2-node.x*state.view.scale;state.view.y=(rect.height-50)/2-node.y*state.view.scale;}

function bindUI() {
  ["model-filter","case-filter","arm-filter"].forEach(id=>$(id).addEventListener("change",refreshTrajectoryOptions));
  $("snapshot-filter").addEventListener("change",()=>{refreshVisibility();fitGraph();});
  $("show-stubs").addEventListener("change",()=>{refreshVisibility();fitGraph();});
  $("trajectory-select").addEventListener("change",e=>{selectTrajectory(e.target.value);fitGraph();});
  $("play-button").addEventListener("click",togglePlay);$("prev-button").addEventListener("click",()=>advance(-1));$("next-button").addEventListener("click",()=>advance(1));
  $("event-slider").addEventListener("input",e=>{state.playing=false;$("play-button").textContent="▶";state.cursor=Number(e.target.value);state.eventProgress=0;updateTimeline();fitGraph();});
  $("speed-select").addEventListener("change",e=>state.speed=Number(e.target.value));
  $("fit-button").addEventListener("click",()=>fitGraph());$("layout-button").addEventListener("click",()=>{initializePositions(state.nodes);fitGraph();});
  const find=()=>{const query=$("page-search").value.trim().toLowerCase();const candidates=state.visibleNodes;const exact=candidates.find(n=>n.title.toLowerCase()===query);const node=exact||candidates.find(n=>n.title.toLowerCase().includes(query));if(node)focusNode(node);};
  $("search-button").addEventListener("click",find);$("page-search").addEventListener("keydown",e=>{if(e.key==="Enter")find();});

  canvas.addEventListener("wheel",e=>{e.preventDefault();const rect=canvas.getBoundingClientRect(),sx=e.clientX-rect.left,sy=e.clientY-rect.top,before=screenToWorld(sx,sy);const factor=Math.exp(-e.deltaY*.0012);state.view.scale=Math.max(.05,Math.min(5,state.view.scale*factor));state.view.x=sx-before.x*state.view.scale;state.view.y=sy-before.y*state.view.scale;},{passive:false});
  canvas.addEventListener("pointerdown",e=>{canvas.setPointerCapture(e.pointerId);const node=hitTest(e.clientX,e.clientY);state.pointerStart={x:e.clientX,y:e.clientY};if(node){state.draggingNode=node;node.vx=0;node.vy=0;}else{state.panning=true;state.viewStart={...state.view};}canvas.classList.add("dragging");});
  canvas.addEventListener("pointermove",e=>{if(state.draggingNode){const rect=canvas.getBoundingClientRect(),w=screenToWorld(e.clientX-rect.left,e.clientY-rect.top);state.draggingNode.x=w.x;state.draggingNode.y=w.y;state.layoutRunning=true;}else if(state.panning){state.view.x=state.viewStart.x+(e.clientX-state.pointerStart.x);state.view.y=state.viewStart.y+(e.clientY-state.pointerStart.y);}else{state.hover=hitTest(e.clientX,e.clientY);showTooltip(state.hover,e);canvas.style.cursor=state.hover?"pointer":"grab";}});
  canvas.addEventListener("pointerup",e=>{const moved=state.pointerStart&&Math.hypot(e.clientX-state.pointerStart.x,e.clientY-state.pointerStart.y)>4;if(state.draggingNode&&!moved)showDetails(state.draggingNode);state.draggingNode=null;state.panning=false;canvas.classList.remove("dragging");});
  canvas.addEventListener("pointerleave",()=>{if(!state.draggingNode&&!state.panning){state.hover=null;showTooltip(null);}});
  canvas.addEventListener("dblclick",e=>{const node=hitTest(e.clientX,e.clientY);if(node)focusNode(node);});
  window.addEventListener("resize",()=>{resizeCanvas();fitGraph();});
}

function renderLoop(now) { physicsStep();updatePlayback(now);draw(now);requestAnimationFrame(renderLoop); }

async function boot() {
  try {
    const response=await fetch("data.json",{cache:"no-store"});if(!response.ok)throw new Error(`data.json: HTTP ${response.status}`);
    state.data=await response.json();state.nodes=state.data.graph.nodes;state.edges=state.data.graph.edges;
    buildIndexes();initializePositions(state.nodes);refreshVisibility();resizeCanvas();
    $("trajectory-count").textContent=state.data.trajectories.length.toLocaleString();
    populateSelect("model-filter",state.data.filters.models,"All models");populateSelect("case-filter",state.data.filters.cases,"All cases");populateSelect("arm-filter",state.data.filters.arms,"All arms");populateSelect("snapshot-filter",state.data.filters.snapshot_groups,"All snapshots");
    refreshTrajectoryOptions();const titles=[...new Set(state.nodes.map(n=>n.title))].sort();$("page-titles").replaceChildren(...titles.map(t=>new Option(t)));
    bindUI();setTimeout(()=>fitGraph(),80);requestAnimationFrame(renderLoop);
  } catch(error) { $("fatal").hidden=false;$("fatal-message").textContent=String(error?.stack||error); }
}

boot();
