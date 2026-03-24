DN.forEach(n=>{if(RICH[n.id])Object.assign(n,RICH[n.id])});
// ── CONCEPT PAGES — node ID → relative URL ──────────────────────────────────
const CONCEPT_PAGES={
  peft_methods:'concepts/peft-methods.html',
  alignment:'concepts/alignment.html',
  kv_cache:'concepts/kv-cache.html',
  quantization:'concepts/quantization.html',
  advanced_reasoning:'concepts/advanced-reasoning.html',
  serving:'concepts/serving.html',
  advanced_rag:'concepts/advanced-rag.html',
  safety_tech:'concepts/safety-tech.html',
  evals_practice:'concepts/evals-practice.html',
  benchmarks:'concepts/benchmarks.html',
  agent_frameworks:'concepts/agent-frameworks.html',
  agent_planning:'concepts/agent-planning.html',
  embeddings_topic:'concepts/embeddings.html',
  retrieval_tech:'concepts/retrieval-tech.html',
  llm_internals:'concepts/llm-internals.html',
  transformer_arch:'concepts/transformer-arch.html',
  frontier_models:'concepts/frontier-models.html',
  multi_agent:'concepts/multi-agent.html',
  tool_use:'concepts/tool-use.html',
  mlops:'concepts/mlops.html',
  synthetic_data:'concepts/synthetic-data.html',
  training_tech:'concepts/training-tech.html',
  output_control:'concepts/output-control.html',
  vision_language:'concepts/vision-language.html',
  agent_memory:'concepts/agent-memory.html',
  basic_prompting:'concepts/basic-prompting.html',
  vector_dbs:'concepts/vector-dbs.html',
  programmatic_prompting:'concepts/programmatic-prompting.html',
  reliability:'concepts/reliability.html',
  hardware:'concepts/hardware.html',
  // batch 4
  image_gen:'concepts/image-gen.html',
  ft_tools:'concepts/ft-tools.html',
  open_models:'concepts/open-models.html',
  compound_ai:'concepts/compound-ai.html',
  data_centric:'concepts/data-centric.html',
  monitoring:'concepts/monitoring.html',
  cloud_deploy:'concepts/cloud-deploy.html',
  data_labeling:'concepts/data-labeling.html',
  neural_nets:'concepts/neural-nets.html',
  attention:'concepts/attention.html',
  // batch 5
  chunking:'concepts/chunking.html',
  optimization:'concepts/optimization.html',
  audio_models:'concepts/audio-models.html',
  video_models:'concepts/video-models.html',
  pos_encoding:'concepts/pos-encoding.html',
  data_prep:'concepts/data-prep.html',
  rag_eval:'concepts/rag-eval.html',
  data_ingestion:'concepts/data-ingestion.html',
  state_sessions:'concepts/state-sessions.html',
  post_retrieval:'concepts/post-retrieval.html',
  // batch 6 — all remaining g2 nodes
  unstructured:'concepts/unstructured.html',
  docling:'concepts/docling.html',
  integration_std:'concepts/integration-std.html',
  dev_frameworks:'concepts/dev-frameworks.html',
  data_governance:'concepts/data-governance.html',
  math_foundations:'concepts/math-foundations.html',
  python_ecosystem:'concepts/python-ecosystem.html',
  pytorch_basics:'concepts/pytorch-basics.html',
  regularization:'concepts/regularization.html',
  execution_models:'concepts/execution-models.html',
  traffic_cost:'concepts/traffic-cost.html',
  human_oversight:'concepts/human-oversight.html',
  decision_fwk:'concepts/decision-frameworks.html',
  frontier_layer:'concepts/frontier-implications.html',
  // batch 7 — 20 new deep-dive pages
  mcp:'concepts/mcp.html',
  rlhf:'concepts/rlhf.html',
  dpo:'concepts/dpo.html',
  graphrag:'concepts/graphrag.html',
  scaling_laws:'concepts/scaling-laws.html',
  moe:'concepts/moe.html',
  flash_attn:'concepts/flash-attn.html',
  vllm:'concepts/vllm.html',
  contextual_retrieval:'concepts/contextual-retrieval.html',
  prompt_injection:'concepts/prompt-injection.html',
  red_teaming:'concepts/red-teaming.html',
  tokenization:'concepts/tokenization.html',
  agentic_rag:'concepts/agentic-rag.html',
  llm_judge:'concepts/llm-judge.html',
  langchain:'concepts/langchain.html',
  langgraph:'concepts/langgraph.html',
  litellm:'concepts/litellm.html',
  streaming:'concepts/streaming.html',
  cost_quality_triangle:'concepts/cost-quality-triangle.html',
  lora:'concepts/lora.html',
  // batch 8 — top-level domain overview pages
  rag:'concepts/rag.html',
  agents:'concepts/agents.html',
  finetuning:'concepts/finetuning.html',
  prompting:'concepts/prompting.html',
  transformers_domain:'concepts/transformers.html',
  llms:'concepts/llms.html',
  multimodal:'concepts/multimodal.html',
  sysdesign:'concepts/system-design.html',
  // batch 9 — remaining domain overview pages
  eval:'concepts/eval.html',
  infra:'concepts/infra.html',
  safety:'concepts/safety.html',
  ml_core_domain:'concepts/ml-core.html',
  foundations:'concepts/foundations.html',
  code_assist:'concepts/code-assist.html',
  // batch 10 — meta cluster hubs + app/building domains
  meta_building:'concepts/meta-building.html',
  meta_applications:'concepts/meta-applications.html',
  meta_foundations:'concepts/meta-foundations.html',
  meta_production:'concepts/meta-production.html',
  meta_governance:'concepts/meta-governance.html',
  data_eng:'concepts/data-eng.html',
  prod_eng:'concepts/prod-eng.html',
  framework_tools:'concepts/framework-tools.html',
  rag_systems:'concepts/rag-systems.html',
  doc_processing:'concepts/doc-processing.html',
  structured_output_app:'concepts/structured-output-app.html',
  database_query:'concepts/database-query.html',
  voice_agents:'concepts/voice-agents.html',
  // batch 11 — g3 leaf nodes: math / python ecosystem / pytorch / neural nets / optimization / training tech / regularization
  linear_algebra:'concepts/linear-algebra.html',
  calculus:'concepts/calculus.html',
  probability:'concepts/probability.html',
  numpy:'concepts/numpy.html',
  pandas:'concepts/pandas.html',
  matplotlib:'concepts/matplotlib.html',
  pytorch_t:'concepts/pytorch-tensors.html',
  autograd:'concepts/autograd.html',
  hf_datasets:'concepts/hf-datasets.html',
  backprop:'concepts/backprop.html',
  activations:'concepts/activations.html',
  batch_norm:'concepts/layer-norm.html',
  adam:'concepts/adam.html',
  lr_schedule:'concepts/lr-schedule.html',
  weight_init:'concepts/weight-init.html',
  mixed_prec:'concepts/mixed-precision.html',
  grad_ckpt:'concepts/grad-checkpointing.html',
  deepspeed:'concepts/deepspeed.html',
  dropout:'concepts/dropout.html',
  weight_decay:'concepts/weight-decay.html',
};
let N=JSON.parse(JSON.stringify(DN)),X=JSON.parse(JSON.stringify(DX)),e=false,s=null;
function ifs(){const c=document.getElementById('starfield'),x=c.getContext('2d');c.width=window.innerWidth;c.height=window.innerHeight;const t=[];for(let i=0;i<100;i++)t.push({x:Math.random()*c.width,y:Math.random()*c.height,r:Math.random()*1.5,v:Math.random()*0.3+0.1});function a(){x.fillStyle='#030a1a';x.fillRect(0,0,c.width,c.height);x.fillStyle='#a78bfa';t.forEach(d=>{d.r+=d.v*0.01;if(d.r>2.5)d.r=0;x.beginPath();x.arc(d.x,d.y,d.r,0,Math.PI*2);x.fill()});requestAnimationFrame(a)}a();window.addEventListener('resize',()=>{c.width=window.innerWidth;c.height=window.innerHeight})}
function hc(cd){
  // HTML-escape raw code first
  let s=cd.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
  // Segment-based approach: scan for strings/comments, highlight keywords only in gaps.
  // No placeholder characters needed — nothing extra can bleed into the output.
  const out=[];
  const re=/("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\'|#[^\n]*|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g;
  const kw=(seg)=>{
    seg=seg.replace(/\b(from|import|class|def|async|await|return|for|if|elif|else|in|not|and|or|with|as|try|except|print|True|False|None|pass|raise|yield|lambda|while|break|continue|self|super)\b/g,'<span style="color:#c084fc">$1</span>');
    seg=seg.replace(/\b(\d+\.?\d*)\b/g,'<span style="color:#fb923c">$1</span>');
    return seg;
  };
  let last=0,m;
  while((m=re.exec(s))!==null){
    out.push(kw(s.slice(last,m.index)));
    const t=m[0];
    if(t[0]==='#')out.push('<span style="color:#6b7280;font-style:italic">'+t+'</span>');
    else out.push('<span style="color:#34d399">'+t+'</span>');
    last=m.index+t.length;
  }
  out.push(kw(s.slice(last)));
  return out.join('');
}
function bl(){
  const c=document.getElementById('legend-items');c.innerHTML='';
  // 5 meta-clusters (g:1, meta:true)
  const clusters=N.filter(n=>n.g===1&&n.meta);
  const openStates={};
  clusters.forEach(cl=>{
    openStates[cl.id]=false;
    // cluster header row
    const wrap=document.createElement('div');
    const hdr=document.createElement('div');
    hdr.className='legend-cluster';
    hdr.innerHTML=`<span style="font-size:9px;color:rgba(255,255,255,0.4)" class="legend-cluster-arrow">▶</span><span>${cl.label}</span>`;
    // domain list (collapsed by default)
    const domWrap=document.createElement('div');
    domWrap.className='legend-domains collapsed';
    // fill domains — g:1 non-meta children of this cluster
    const domains=N.filter(n=>n.g===1&&!n.meta&&n.p===cl.id);
    domains.forEach(n=>{
      const t=CATS[n.cat]||CATS.custom;
      const item=document.createElement('div');
      item.className='legend-item';
      item.innerHTML=`<div class="legend-dot" style="background:${t.c}"></div><span>${n.label}</span>`;
      item.dataset.clusterId=cl.id;item.dataset.domainId=n.id;
      item.addEventListener('click',(e)=>{
        e.stopPropagation();
        const clNode=N.find(x=>x.id===e.currentTarget.dataset.clusterId);
        const dNode=N.find(x=>x.id===e.currentTarget.dataset.domainId);
        if(!clNode||!dNode)return;
        const rootNode=N.find(x=>x.g===0);
        if(rootNode)rootNode.expanded=true;
        clNode.expanded=true;dNode.expanded=true;
        rg();
        sn(dNode);sp(dNode);
        // pan to the domain node after simulation settles
        setTimeout(()=>{
          if(!zb||dNode.x===undefined)return;
          const w=window.innerWidth,h=window.innerHeight-60;
          const t=d3.zoomIdentity.translate(w/2-dNode.x,h/2-dNode.y);
          svg.transition().duration(600).call(zb.transform,t);
        },600);
      });
      domWrap.appendChild(item);
    });
    hdr.addEventListener('click',()=>{
      openStates[cl.id]=!openStates[cl.id];
      domWrap.classList.toggle('collapsed',!openStates[cl.id]);
      hdr.querySelector('.legend-cluster-arrow').classList.toggle('open',openStates[cl.id]);
      // also expand/collapse the cluster in the graph
      const rootNode=N.find(x=>x.g===0);
      if(rootNode)rootNode.expanded=true;
      cl.expanded=openStates[cl.id];
      rg();
    });
    wrap.appendChild(hdr);wrap.appendChild(domWrap);c.appendChild(wrap);
  });
}
let sim=null,svg,gc,tr=d3.zoomIdentity,zb=null;
window._expandNode=function(nodeId){const node=N.find(n=>n.id===nodeId);if(!node)return;node.expanded=!node.expanded;if(typeof gtag==='function')gtag('event','node_expand',{node_id:node.id,node_label:node.label,action:node.expanded?'expand':'collapse'});rg();};
document.addEventListener('click',function(e){const chip=e.target.closest?e.target.closest('.expand-chip'):null;if(!chip)return;const nodeId=chip.getAttribute('data-nodeid');if(!nodeId)return;e.stopPropagation();window._expandNode(nodeId);},true);
// Context menu
let _ctxNode=null;
const ctxMenu=document.getElementById('ctx-menu');
function showCtxMenu(e,d){
  _ctxNode=d;
  const hasChildren=N.filter(n=>n.p===d.id).length>0;
  document.getElementById('ctx-expand').style.display=hasChildren?'flex':'none';
  document.getElementById('ctx-expand-label').textContent=d.expanded?'Collapse':'Expand';
  const x=Math.min(e.clientX,window.innerWidth-190),y=Math.min(e.clientY,window.innerHeight-140);
  ctxMenu.style.left=x+'px';ctxMenu.style.top=y+'px';ctxMenu.classList.add('visible');
}
document.addEventListener('click',()=>ctxMenu.classList.remove('visible'));
document.addEventListener('contextmenu',e=>{if(!e.target.closest('.node-card'))ctxMenu.classList.remove('visible');});
document.getElementById('ctx-expand').addEventListener('click',()=>{if(_ctxNode){window._expandNode(_ctxNode.id);}ctxMenu.classList.remove('visible');});
document.getElementById('ctx-open').addEventListener('click',()=>{if(_ctxNode){sn(_ctxNode);sp(_ctxNode);}ctxMenu.classList.remove('visible');});
document.getElementById('ctx-copy').addEventListener('click',()=>{if(_ctxNode){navigator.clipboard?.writeText(_ctxNode.label).catch(()=>{});}ctxMenu.classList.remove('visible');});
function ig(){gc=d3.select('#graph-container');svg=d3.select('#graph');const w=window.innerWidth,h=window.innerHeight-60;sim=d3.forceSimulation(N).force('link',d3.forceLink(X.map(l=>({source:N.find(n=>n.id===l.s),target:N.find(n=>n.id===l.t)}))).id(d=>d.id).distance(100).strength(0.3)).force('charge',d3.forceManyBody().strength(-500).distanceMax(500)).force('center',d3.forceCenter(w/2,h/2)).force('collide',d3.forceCollide(d=>(d.meta?155:SZ[Math.min(d.g,3)].w)/2+20)).on('tick',ug);svg.append('defs').html(`<marker id="arrowhead" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><polygon points="0 0, 10 3, 0 6" fill="rgba(255,255,255,0.15)"/></marker><marker id="arrowhead-x" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><polygon points="0 0, 10 3, 0 6" fill="rgba(200,200,200,0.2)"/></marker>`);svg.append('g').attr('class','links');svg.append('g').attr('class','nodes');zb=d3.zoom().on('zoom',e=>{tr=e.transform;svg.select('.links').attr('transform',tr);svg.select('.nodes').attr('transform',tr)});svg.call(zb);window.addEventListener('resize',()=>{sim.force('center',d3.forceCenter(window.innerWidth/2,(window.innerHeight-60)/2))})}
function rg(){const vn=N.filter(n=>{if(n.g===0)return true;let p=n.p;while(p){const pn=N.find(x=>x.id===p);if(!pn)return false;if(!pn.expanded)return false;p=pn.p}return true});const vid=new Set(vn.map(n=>n.id)),vl=X.filter(l=>vid.has(l.s)&&vid.has(l.t)),ld=vl.map(l=>({source:N.find(n=>n.id===l.s),target:N.find(n=>n.id===l.t),xlink:true}));N.filter(n=>n.g>0).forEach(n=>{if(n.p){const pn=N.find(x=>x.id===n.p);if(pn&&vid.has(n.id)&&vid.has(pn.id))ld.push({source:pn,target:n,xlink:false})}});sim.nodes(vn);sim.force('link').links(ld);const lks=svg.select('.links').selectAll('.link').data(ld,(d,i)=>i);lks.exit().remove();lks.enter().append('line').attr('class',d=>`link ${d.xlink?'xlink':''}`).merge(lks);const ns=svg.select('.nodes').selectAll('.node-card').data(vn,d=>d.id);ns.exit().remove();const ne=ns.enter().append('g').attr('class',d=>`node-card ${d.id===s?.id?'selected':''}`).attr('data-id',d=>d.id).attr('data-g',d=>d.g).attr('data-meta',d=>d.meta?'true':null).on('click',(e,d)=>{e.stopPropagation();sn(d);sp(d)}).on('dblclick',(e,d)=>{e.stopPropagation();d.expanded=!d.expanded;if(typeof gtag==='function')gtag('event','node_expand',{node_id:d.id,node_label:d.label,action:d.expanded?'expand':'collapse'});rg()}).on('contextmenu',(e,d)=>{e.preventDefault();e.stopPropagation();showCtxMenu(e.sourceEvent||e,d);}).call(d3.drag().on('start',dgs).on('drag',dg).on('end',dge));const sz=d=>d.meta?{w:155,h:62}:SZ[Math.min(d.g,3)];ne.append('title').text(d=>{const cc=N.filter(n=>n.p===d.id&&n.g<4).length;return cc>0?'Click ▸ chip to expand · Click node for details':'Click to open details'});ne.append('rect').attr('class','node-rect').attr('x',d=>-sz(d).w/2).attr('y',d=>-sz(d).h/2).attr('width',d=>sz(d).w).attr('height',d=>sz(d).h).attr('rx',4).attr('fill',d=>CATS[d.cat].c).attr('stroke',d=>CATS[d.cat].c);ne.append('foreignObject').attr('x',d=>-sz(d).w/2+4).attr('y',d=>-sz(d).h/2+4).attr('width',d=>sz(d).w-8).attr('height',d=>sz(d).h-8).append('xhtml:div').style('padding','4px').style('display','flex').style('flex-direction','column').style('align-items','center').style('justify-content','center').style('height','100%').style('text-align','center').html(d=>{let h=`<div class="node-label">${d.label}</div>`;if(d.meta){const cnt=N.filter(n=>n.p===d.id).length;h+=`<div class="node-category">${cnt} domains</div>`;}else if(d.g<=2)h+=`<div class="node-category">${CATS[d.cat].n}</div>`;const childCount=N.filter(n=>n.p===d.id&&n.g<4).length;if(childCount>0){const label=d.g===0||d.meta?`▸ ${childCount} domains`:d.g===1?`▸ ${childCount} topics`:`▸ ${childCount} tools`;h+=`<div class="expand-chip" data-nodeid="${d.id}">${label}</div>`;}if(d.code||d.diag||d.tip)h+=`<div class="rich-dot" title="Has code &amp; diagrams — click to explore">◆</div>`;return h});sim.alpha(1).restart()}
function ug(){svg.selectAll('.link').attr('x1',d=>d.source.x).attr('y1',d=>d.source.y).attr('x2',d=>d.target.x).attr('y2',d=>d.target.y);svg.selectAll('.node-card').attr('transform',d=>`translate(${d.x},${d.y})`)}
function dgs(e,d){if(!e.active)sim.alphaTarget(0.3).restart();d.fx=d.x;d.fy=d.y}
function dg(e,d){d.fx=e.x;d.fy=e.y}
function dge(e,d){if(!e.active)sim.alphaTarget(0);d.fx=null;d.fy=null}
function sn(d){s=d;document.querySelectorAll('.node-card').forEach(el=>el.classList.remove('selected'));document.querySelector(`[data-id="${d.id}"]`)?.classList.add('selected');hlLegend(d.id);}
function hlLegend(nodeId){let node=N.find(n=>n.id===nodeId);while(node&&node.g>1){node=N.find(n=>n.id===node.p);}document.querySelectorAll('.legend-item').forEach(el=>el.classList.toggle('active',el.dataset.domainId===node?.id));}
function sp(d){
  // GA4 — track which node was opened
  if(typeof gtag==='function'){
    gtag('event','node_open',{
      node_id:d.id,
      node_label:d.label,
      node_depth:d.g,
      node_domain:d.cat,
      has_rich:!!(d.use||d.code)
    });
  }
  const p=document.getElementById('panel'),c=CATS[d.cat]||CATS.custom;
  // BREADCRUMB — build parent chain for back navigation
  const bcEl=document.getElementById('panel-breadcrumb');
  const chain=[];let cur=d;while(cur){chain.unshift(cur);if(cur.p)cur=N.find(n=>n.id===cur.p);else break;}
  bcEl.innerHTML=chain.map((n,i)=>{
    const isCurrent=i===chain.length-1;
    return `${i>0?'<span class="bc-sep">›</span>':''}<span class="bc-item${isCurrent?' bc-current':''}" data-bcid="${n.id}">${n.label}</span>`;
  }).join('');
  bcEl.querySelectorAll('.bc-item:not(.bc-current)').forEach(el=>el.addEventListener('click',()=>{const t=N.find(n=>n.id===el.dataset.bcid);if(t){sn(t);sp(t)}}));
  // BACK BUTTON — visible whenever there is a parent node
  const backBtn=document.getElementById('panel-back');
  if(d.p){
    const parentNode=N.find(n=>n.id===d.p);
    backBtn.textContent=`← ${parentNode?parentNode.label:'Back'}`;
    backBtn.classList.add('visible');
    backBtn.onclick=()=>{if(parentNode){sn(parentNode);sp(parentNode)}};
  } else {
    backBtn.classList.remove('visible');
    backBtn.onclick=null;
  }
  // DEPTH LABEL
  const depthLabels={0:'KNOWLEDGE MAP',1:'DOMAIN',2:'TOPIC',3:'TOOL / TECHNIQUE'};
  document.getElementById('panel-depth-label').textContent=d.meta?'CLUSTER':depthLabels[d.g]||'NODE';
  // CATEGORY BADGE
  const badge=document.getElementById('panel-cat');
  badge.style.background=`${c.c}18`;badge.style.color=c.c;
  badge.style.borderColor=`${c.c}35`;badge.textContent=c.n;
  // TITLE + DESCRIPTION
  document.getElementById('panel-title').textContent=d.label;
  document.getElementById('panel-desc').textContent=d.desc||'';
  // DEEP DIVE LINK
  const ddEl=document.getElementById('ps-deepdive');
  if(CONCEPT_PAGES[d.id]){ddEl.href=CONCEPT_PAGES[d.id];ddEl.style.display='inline-flex';}
  else ddEl.style.display='none';
  // WHY IT MATTERS
  if(d.use){const whyEl=document.getElementById('ps-why-text');whyEl.innerHTML='';d.use.split('\n\n').forEach(para=>{const p=document.createElement('p');p.textContent=para;whyEl.appendChild(p);});document.getElementById('ps-why').style.display='block'}
  else document.getElementById('ps-why').style.display='none';
  // DIAGRAM
  if(d.diag){document.getElementById('diag-block').textContent=d.diag;document.getElementById('ps-diag').style.display='block'}
  else document.getElementById('ps-diag').style.display='none';
  // LEARNING PATH — use PATHS for g:1 domain nodes, otherwise show child pills
  const path=PATHS[d.id];
  const learnEl=document.getElementById('ps-learn-pills');
  const learnSection=document.getElementById('ps-learn');
  const learnTitle=learnSection.querySelector('.panel-section-title');
  if(path&&path.length>0){
    learnTitle.textContent='Learning path';
    learnEl.innerHTML=path.map((step,i)=>{const node=N.find(n=>n.id===step.id);if(!node)return '';return `<div class="path-item" data-target="${step.id}"><div class="path-num">${i+1}</div><div><div class="path-node-name">${node.label}</div><div class="path-note">${step.note}</div></div></div>`;}).join('');
    learnSection.style.display='block';
  } else {
    const children=N.filter(n=>n.p===d.id);
    if(children.length>0){
      learnTitle.textContent='What to learn';
      learnEl.innerHTML=children.map(ch=>`<div class="panel-pill" data-target="${ch.id}">${ch.label}</div>`).join('');
      learnSection.style.display='block';
    } else learnSection.style.display='none';
  }
  learnEl.querySelectorAll('[data-target]').forEach(el=>el.addEventListener('click',()=>{const t=N.find(n=>n.id===el.dataset.target);if(t){sn(t);sp(t)}}));
  // KEY TOOLS — only for g:1 domain nodes (g:2 topics already show their g:3 children above)
  if(d.g===1){
    const getAllDesc=(id)=>{const kids=N.filter(n=>n.p===id);return kids.flatMap(k=>[k,...getAllDesc(k.id)])};
    const tools=getAllDesc(d.id).filter(n=>n.g===3).slice(0,14);
    if(tools.length>0){
      const toolsEl=document.getElementById('ps-tools-pills');
      toolsEl.innerHTML=tools.map(t=>`<div class="panel-pill pill-tool" data-target="${t.id}">${t.label}</div>`).join('');
      toolsEl.querySelectorAll('.panel-pill').forEach(pill=>pill.addEventListener('click',()=>{const t=N.find(n=>n.id===pill.dataset.target);if(t){sn(t);sp(t)}}));
      document.getElementById('ps-tools-sec').style.display='block';
    } else document.getElementById('ps-tools-sec').style.display='none';
  } else document.getElementById('ps-tools-sec').style.display='none';
  // PYTHON EXAMPLE
  if(d.code){const cb=document.getElementById('code-block');cb.innerHTML=hc(d.code);document.getElementById('ps-code').style.display='block';const cpBtn=document.getElementById('copy-code-btn');cpBtn.onclick=()=>{const raw=d.code;const ta=document.createElement('textarea');ta.value=raw;ta.style.cssText='position:fixed;top:0;left:0;opacity:0;pointer-events:none';document.body.appendChild(ta);ta.focus();ta.select();document.execCommand('copy');document.body.removeChild(ta);cpBtn.textContent='✓ copied!';setTimeout(()=>{cpBtn.textContent='📋 copy'},2000)};}
  else document.getElementById('ps-code').style.display='none';
  // PRO TIP
  if(d.tip){const tipEl=document.getElementById('ps-tip-text');tipEl.innerHTML='';d.tip.split('\n\n').forEach(para=>{const p=document.createElement('p');p.style.whiteSpace='pre-line';p.textContent=para;tipEl.appendChild(p);});document.getElementById('ps-tip').style.display='block'}
  else document.getElementById('ps-tip').style.display='none';
  // FURTHER READING
  if(d.refs && d.refs.length){
    const rl=document.getElementById('ps-refs-list');
    rl.innerHTML=d.refs.map(r=>`<a href="${r.url}?from=${d.id}" target="_blank" class="ref-link">→ ${r.label} ↗</a>`).join('');
    document.getElementById('ps-refs').style.display='block';
  } else document.getElementById('ps-refs').style.display='none';
  // KEY QUESTIONS
  if(d.questions){
    const qBody=document.getElementById('ps-questions-body');
    qBody.innerHTML='';
    [{key:'leader',label:'For Leaders'},{key:'pm',label:'For PMs'},{key:'eng',label:'For Engineers'}].forEach(r=>{
      const qs=d.questions[r.key];
      if(!qs||!qs.length)return;
      const grp=document.createElement('div');grp.className='qs-group';
      const lbl=document.createElement('div');lbl.className='qs-role';lbl.textContent=r.label;
      const ul=document.createElement('ul');ul.className='qs-list';
      qs.forEach(q=>{const li=document.createElement('li');li.textContent=q;ul.appendChild(li);});
      grp.appendChild(lbl);grp.appendChild(ul);qBody.appendChild(grp);
    });
    document.getElementById('ps-questions').style.display='block';
  } else document.getElementById('ps-questions').style.display='none';
  // CONNECTED TOPICS — cross-links
  const xl=X.filter(l=>l.s===d.id||l.t===d.id);
  if(xl.length>0){
    const linksEl=document.getElementById('panel-links');
    linksEl.innerHTML=xl.map(l=>{const isSrc=l.s===d.id,other=N.find(n=>n.id===(isSrc?l.t:l.s));if(!other)return '';const arrow=isSrc?'→':'←';return `<div class="panel-pill" data-target="${other.id}" title="${arrow} ${l.label}">${other.label}</div>`}).join('');
    linksEl.querySelectorAll('.panel-pill').forEach(pill=>pill.addEventListener('click',()=>{const t=N.find(n=>n.id===pill.dataset.target);if(t){sn(t);sp(t)}}));
    document.getElementById('ps-links').style.display='block';
  } else document.getElementById('ps-links').style.display='none';
  // Restore saved panel width before opening so graph-container padding is correct
  const _sw=localStorage.getItem('panel_width');if(_sw){const _w=parseInt(_sw);if(_w>=320&&_w<=window.innerWidth*0.85)p.style.width=_w+'px';}
  p.classList.add('open');if(window.innerWidth>768)document.getElementById('graph-container').style.paddingRight=p.offsetWidth+'px';
  // location.hash works on file:// and https:// alike
  location.hash=d.id;
}
document.getElementById('panel-close').addEventListener('click',()=>{document.getElementById('panel').classList.remove('open');document.getElementById('graph-container').style.paddingRight='0';document.body.classList.remove('focus-mode');s=null;svg.selectAll('.node-card').classed('selected',false);try{history.replaceState(null,'',location.pathname+location.search);}catch(e){location.hash='';}});
document.getElementById('panel-focus').addEventListener('click',()=>{const fm=document.body.classList.toggle('focus-mode');document.getElementById('panel-focus').title=fm?'Back to map':'Focus mode — expand to full screen';document.getElementById('panel-focus').textContent=fm?'⊠':'⛶';if(!fm&&window.innerWidth>768)document.getElementById('graph-container').style.paddingRight=document.getElementById('panel').offsetWidth+'px';});
// [Clusters] — show only the 5 meta-cluster nodes (collapse everything, expand root only)
document.getElementById('btn-domains').addEventListener('click',()=>{N.forEach(n=>n.expanded=false);const r=N.find(n=>n.g===0);if(r)r.expanded=true;rg();if(typeof gtag==='function')gtag('event','toolbar_click',{button:'clusters_only'})});
// [+Domains] — show meta-clusters + domain nodes (expand root + all meta-clusters)
document.getElementById('btn-topics').addEventListener('click',()=>{N.forEach(n=>n.expanded=false);const r=N.find(n=>n.g===0);if(r)r.expanded=true;N.filter(n=>n.meta).forEach(n=>n.expanded=true);rg();if(typeof gtag==='function')gtag('event','toolbar_click',{button:'show_domains'})});
// [+Topics] — show everything down to g:2 topics (expand root + meta + all g:1 domains)
document.getElementById('btn-tools').addEventListener('click',()=>{N.forEach(n=>n.expanded=false);const r=N.find(n=>n.g===0);if(r)r.expanded=true;N.filter(n=>n.meta||n.g===1).forEach(n=>n.expanded=true);rg();if(typeof gtag==='function')gtag('event','toolbar_click',{button:'show_topics'})});
// [+All] — expand everything
document.getElementById('btn-collapse').addEventListener('click',()=>{N.forEach(n=>n.expanded=true);rg();if(typeof gtag==='function')gtag('event','toolbar_click',{button:'expand_all'})});
document.getElementById('btn-fit').addEventListener('click',()=>{
  // match same visibility logic as rg()
  const nodes=N.filter(n=>{if(n.g===0)return true;let p=n.p;while(p){const pn=N.find(x=>x.id===p);if(!pn)return false;if(!pn.expanded)return false;p=pn.p}return true});
  if(!nodes.length||!zb)return;
  const xs=nodes.map(n=>n.x),ys=nodes.map(n=>n.y);
  const mnx=Math.min(...xs),mxx=Math.max(...xs),mny=Math.min(...ys),mxy=Math.max(...ys);
  const pad=60,H=window.innerHeight-60;
  // subtract panel width if open so graph fits in the visible area
  const panel=document.getElementById('panel');
  const panelW=panel.classList.contains('open')?panel.offsetWidth:0;
  const W=window.innerWidth-panelW;
  const sc=Math.min((W-pad*2)/(mxx-mnx||1),(H-pad*2)/(mxy-mny||1))*0.9;
  // centre within the visible area (left of panel), not the full window
  const tx=(W/2)-(mnx+mxx)/2*sc,ty=H/2-(mny+mxy)/2*sc;
  // use stored zb — not a new d3.zoom() — so pan/scroll stays in sync
  svg.transition().duration(750).call(zb.transform,d3.zoomIdentity.translate(tx,ty).scale(sc));
});
document.getElementById('btn-reset').addEventListener('click',()=>{N=JSON.parse(JSON.stringify(DN));DN.forEach(n=>{if(RICH[n.id])Object.assign(n,RICH[n.id])});X=JSON.parse(JSON.stringify(DX));sim.nodes(N);bl();rg()});
document.getElementById('btn-edit').addEventListener('click',()=>{e=!e;document.getElementById('btn-add').classList.toggle('hidden');document.getElementById('btn-link').classList.toggle('hidden');document.getElementById('btn-edit').textContent=e?'[✏ Edit ✓]':'[✏ Edit]'});
document.getElementById('btn-export').addEventListener('click',()=>{const d={N,X},j=JSON.stringify(d,null,2),b=new Blob([j],{type:'application/json'}),u=URL.createObjectURL(b),a=document.createElement('a');a.href=u;a.download='mindmap-export.json';a.click();URL.revokeObjectURL(u)});
document.getElementById('btn-import').addEventListener('click',()=>{const i=document.createElement('input');i.type='file';i.accept='application/json';i.addEventListener('change',(e)=>{const f=e.target.files[0],r=new FileReader();r.onload=(e)=>{try{const d=JSON.parse(e.target.result);N=d.N||DN;X=d.X||DX;rg();bl()}catch(e){alert('Invalid JSON')}};r.readAsText(f)});i.click()});
document.getElementById('topbar-search').addEventListener('input',(e)=>{const q=e.target.value.toLowerCase();document.querySelectorAll('.node-card').forEach(el=>el.classList.remove('search-highlight'));if(!q){rg();return}const m=N.filter(n=>n.label.toLowerCase().includes(q)||n.id.includes(q));if(m.length===0)return;const mid=new Set(m.map(n=>n.id));if(m.length<=8){const te=new Set;m.forEach(m=>{let p=m.p;while(p){te.add(p);const pn=N.find(x=>x.id===p);if(!pn)break;p=pn.p}});N.forEach(n=>{if(te.has(n.id))n.expanded=true})}rg();setTimeout(()=>{document.querySelectorAll('.node-card').forEach(el=>{if(mid.has(el.dataset.id))el.classList.add('search-highlight')})},100)});
// Panel resize drag handle
(function(){
  const handle=document.getElementById('panel-resize');
  const panel=document.getElementById('panel');
  const gc=document.getElementById('graph-container');
  const SAVED_KEY='panel_width';
  // Restore saved width
  const saved=localStorage.getItem(SAVED_KEY);
  if(saved){const w=parseInt(saved);if(w>=320&&w<=window.innerWidth*0.85){panel.style.width=w+'px';}}
  function syncGC(){if(window.innerWidth>768)gc.style.paddingRight=panel.classList.contains('open')?panel.offsetWidth+'px':'0';}
  let dragging=false, startX=0, startW=0;
  handle.addEventListener('mousedown',(e)=>{
    e.preventDefault();
    dragging=true; startX=e.clientX; startW=panel.offsetWidth;
    handle.classList.add('dragging');
    panel.classList.add('resizing');
    document.body.style.userSelect='none';
    document.body.style.cursor='col-resize';
  });
  document.addEventListener('mousemove',(e)=>{
    if(!dragging)return;
    const delta=startX-e.clientX;
    const newW=Math.min(Math.max(startW+delta,320),window.innerWidth*0.85);
    panel.style.width=newW+'px';
    syncGC();
  });
  document.addEventListener('mouseup',()=>{
    if(!dragging)return;
    dragging=false;
    handle.classList.remove('dragging');
    panel.classList.remove('resizing');
    document.body.style.userSelect='';
    document.body.style.cursor='';
    localStorage.setItem(SAVED_KEY,panel.offsetWidth);
    syncGC();
  });
})();
ifs();ig();bl();rg();

// ── SHARE BUTTON — copy direct link to node ──────────────────────────────
document.getElementById('panel-share').addEventListener('click',()=>{
  const url=location.href;
  navigator.clipboard?.writeText(url).then(()=>{
    const t=document.getElementById('share-toast');
    t.classList.add('visible');
    setTimeout(()=>t.classList.remove('visible'),2200);
  }).catch(()=>{});
});

// ── DEEP LINK — open node from URL hash on page load ─────────────────────
(function(){
  const hashId=window.location.hash.replace('#','');
  if(!hashId)return;
  // shared link: skip the welcome overlay so the panel isn't buried behind it
  const overlay=document.getElementById('welcome-overlay');
  if(overlay){overlay.classList.add('hidden');}
  try{localStorage.setItem('gmwelcome','1');}catch(e){}
  // also hide the hint bar immediately
  const hintBar=document.getElementById('hint-bar');
  if(hintBar){hintBar.classList.add('hidden');}
  try{localStorage.setItem('gmhint','1');}catch(e){}
  function expandParentsFor(nodeId){
    let node=N.find(n=>n.id===nodeId);
    while(node&&node.p){const par=N.find(n=>n.id===node.p);if(par)par.expanded=true;node=par;}
  }
  function tryOpen(){
    const d=N.find(n=>n.id===hashId);
    if(!d)return;
    expandParentsFor(d.id);
    rg();
    // give the simulation a beat to re-layout after expanding parents
    setTimeout(()=>{sn(d);sp(d);},400);
  }
  // wait for initial simulation warm-up, then open
  setTimeout(tryOpen,800);
})();

// Welcome overlay — shown on first visit, auto-demo after dismiss
(function(){
  var overlay = document.getElementById('welcome-overlay');
  if (!overlay) return;
  var seen = false;
  try { seen = !!localStorage.getItem('gmwelcome'); } catch(e) {}
  if (seen) { overlay.style.display = 'none'; return; }
  var btn = document.getElementById('welcome-start');
  if (!btn) return;
  btn.addEventListener('click', function() {
    overlay.style.opacity = '0';
    overlay.style.transition = 'opacity 0.4s';
    setTimeout(function() { overlay.style.display = 'none'; }, 450);
    try { localStorage.setItem('gmwelcome', '1'); } catch(e) {}
  });
})();
// Onboarding hint bar — stays until user explicitly dismisses or clicks a node
(function(){
  const bar=document.getElementById('hint-bar');
  const dismiss=()=>{bar.classList.add('hidden');document.body.classList.add('explored');try{localStorage.setItem('gmhint','1')}catch(e){}};
  try{if(localStorage.getItem('gmhint')){bar.classList.add('hidden');document.body.classList.add('explored')}}catch(e){}
  document.getElementById('hint-close').addEventListener('click',dismiss);
  // Dismiss and stop invite pulse on first node click
  document.addEventListener('click',function onFirst(e){
    if(e.target.closest('.node-card')){dismiss();document.removeEventListener('click',onFirst,true)}
  },true);
})();

// ── SEARCH ───────────────────────────────────────────────────────────────────
(function(){
  const input   = document.getElementById('topbar-search');
  const counter = document.getElementById('search-count');
  let preSrch   = null; // snapshot of expansion state before search
  let gaTimer   = null; // debounce GA4 event

  function expandParents(nodeId){
    let node = N.find(n=>n.id===nodeId);
    while(node && node.p){
      const parent = N.find(n=>n.id===node.p);
      if(parent) parent.expanded=true;
      node=parent;
    }
  }

  function applyClasses(matchIds){
    svg.selectAll('.node-card').each(function(d){
      const el=d3.select(this);
      const hit=matchIds.has(d.id);
      el.classed('search-match', hit).classed('search-fade', !hit);
    });
  }

  function clearSearch(){
    input.classList.remove('no-results');
    counter.textContent='';
    // restore pre-search expansion state
    if(preSrch){
      N.forEach(n=>{ if(n.id in preSrch) n.expanded=preSrch[n.id]; });
      preSrch=null;
    }
    rg();
    svg.selectAll('.node-card').classed('search-match',false).classed('search-fade',false);
  }

  input.addEventListener('input',()=>{
    const raw=input.value.trim().toLowerCase();

    if(!raw){ clearSearch(); return; }

    // snapshot expansion state once on first keypress
    if(!preSrch){
      preSrch=N.reduce((acc,n)=>{acc[n.id]=!!n.expanded;return acc;},{});
    }

    // find matches across ALL nodes (label, desc, category name)
    const matches=N.filter(n=>{
      if(n.g===0||n.meta) return false; // skip root & meta-clusters
      const label=(n.label||'').toLowerCase();
      const desc=(n.desc||'').toLowerCase();
      const catName=(CATS[n.cat]?.n||'').toLowerCase();
      return label.includes(raw)||desc.includes(raw)||catName.includes(raw);
    });

    if(matches.length===0){
      input.classList.add('no-results');
      counter.textContent='no results';
      rg();
      svg.selectAll('.node-card').classed('search-match',false).classed('search-fade',true);
      return;
    }

    input.classList.remove('no-results');
    counter.textContent=matches.length+' match'+(matches.length===1?'':'es');

    // expand parent chains so every match is visible
    matches.forEach(n=>expandParents(n.id));
    rg(); // rebuild DOM with newly-visible nodes

    // apply highlight / fade after DOM rebuild
    const matchIds=new Set(matches.map(n=>n.id));
    applyClasses(matchIds);

    // GA4 — debounced so we don't spam on every keystroke
    clearTimeout(gaTimer);
    gaTimer=setTimeout(()=>{
      if(typeof gtag==='function')
        gtag('event','search',{search_term:raw,result_count:matches.length});
    },800);
  });

  // Escape clears search
  input.addEventListener('keydown',e=>{
    if(e.key==='Escape'){ input.value=''; clearSearch(); input.blur(); }
  });

  // Clicking anywhere on the graph while searching selects + opens panel normally
  // (search-fade nodes have pointer-events:none so only matches are clickable)
})();

// ── Mobile: swipe-down on drag handle to dismiss panel ──────────
(function(){
  const handle=document.getElementById('panel-drag-handle');
  const panel=document.getElementById('panel');
  const closeBtn=document.getElementById('panel-close');
  if(!handle||!panel)return;
  let startY=0,startT=0;
  handle.addEventListener('touchstart',e=>{startY=e.touches[0].clientY;startT=Date.now();},{passive:true});
  handle.addEventListener('touchend',e=>{
    const dy=e.changedTouches[0].clientY-startY;
    const dt=Date.now()-startT;
    if(dy>60&&dt<400)closeBtn.click(); // swipe down > 60px in < 400ms = dismiss
  },{passive:true});
})();
