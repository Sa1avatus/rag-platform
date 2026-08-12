import {FormEvent, useState} from "react";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";
import {useSearchParams} from "react-router-dom";
import {api} from "./api";
const Loading=()=> <div className="skeleton">Loading…</div>;
type DashboardSnapshot={documents:number;chunks:number;recent_indexing_failures:number;active_index:string;retrieval_latency_ms:number|null;reranker_latency_ms:number|null;cache_hit_rate:number|null;embedding:{model:string;revision:string};health:{status:string;components:Array<{name:string;status:string}>}};
export function Dashboard(){const q=useQuery({queryKey:["dashboard"],queryFn:()=>api<DashboardSnapshot>("/v1/admin/dashboard"),refetchInterval:10000});if(q.isLoading)return <Loading/>;if(q.error)return <p role="alert">{q.error.message}</p>;const d=q.data;return <><header><div><h1>Dashboard</h1><p className="lede">Current service identity, dependency health, and bounded operational indicators.</p></div></header>{d&&<><section className="cards"><article><span>Documents</span><strong>{d.documents.toLocaleString()}</strong></article><article><span>Chunks</span><strong>{d.chunks.toLocaleString()}</strong></article><article><span>Indexing failures (24h)</span><strong>{d.recent_indexing_failures}</strong></article><article><span>Retrieval latency</span><strong>{d.retrieval_latency_ms==null?"No samples":`${d.retrieval_latency_ms} ms`}</strong></article><article><span>Reranker latency</span><strong>{d.reranker_latency_ms==null?"No samples":`${d.reranker_latency_ms} ms`}</strong></article><article><span>Cache hit rate</span><strong>{d.cache_hit_rate==null?"No samples":`${(d.cache_hit_rate*100).toFixed(1)}%`}</strong></article></section><section className="cards"><article><span>Overall health</span><strong>{d.health.status}</strong></article><article><span>Embedding model</span><strong>{d.embedding.model}</strong><small>{d.embedding.revision}</small></article><article><span>Active index</span><strong>{d.active_index}</strong></article>{d.health.components.map(component=><article key={component.name}><span>{component.name}</span><strong>{component.status}</strong></article>)}</section></>}</>}
export function Projects(){const client=useQueryClient();const q=useQuery({queryKey:["projects"],queryFn:()=>api<any[]>("/v1/admin/projects")});const tenants=useQuery({queryKey:["tenants"],queryFn:()=>api<Array<{id:string;name:string}>>("/v1/admin/tenants")});const[error,setError]=useState<string>();async function create(e:FormEvent<HTMLFormElement>){e.preventDefault();setError(undefined);const form=e.currentTarget;const f=new FormData(form);try{await api("/v1/admin/projects",{method:"POST",body:JSON.stringify(Object.fromEntries(f))});client.invalidateQueries({queryKey:["projects"]});form.reset()}catch(err){setError(err instanceof Error?err.message:"Failed to create project")}}async function createTenant(e:FormEvent<HTMLFormElement>){e.preventDefault();setError(undefined);const form=e.currentTarget;const f=new FormData(form);try{await api("/v1/admin/tenants",{method:"POST",body:JSON.stringify({name:f.get("tenant_name")})});client.invalidateQueries({queryKey:["tenants"]});form.reset()}catch(err){setError(err instanceof Error?err.message:"Failed to create tenant")}}return <><h1>Projects</h1><section><h2>Create tenant</h2><form className="inline" onSubmit={createTenant}><input name="tenant_name" placeholder="Tenant name" required/><button>Add tenant</button></form></section><section><h2>Create project</h2><form className="inline" onSubmit={create}><select name="tenant_id" required><option value="">Select tenant</option>{tenants.data?.map(t=><option key={t.id} value={t.id}>{t.name}</option>)}</select><input name="slug" placeholder="project-slug" required/><input name="name" placeholder="Project name" required/><button>Create</button></form></section>{error&&<p role="alert">{error}</p>}{q.isLoading?<Loading/>:<table><thead><tr><th>Name</th><th>Slug</th><th>Status</th></tr></thead><tbody>{q.data?.map(p=><tr key={p.id}><td>{p.name}</td><td>{p.slug}</td><td>{p.enabled?"Enabled":"Disabled"}</td></tr>)}</tbody></table>}</>}
export function Collections(){const q=useQuery({queryKey:["collections"],queryFn:()=>api<any[]>("/v1/admin/collections")});return <><h1>Collections</h1>{q.isLoading?<Loading/>:<table><thead><tr><th>Name</th><th>Project</th><th>Strategy</th></tr></thead><tbody>{q.data?.map(c=><tr key={c.id}><td>{c.name}</td><td>{c.project_id}</td><td>{c.settings.chunking_strategy??"recursive_text"}</td></tr>)}</tbody></table>}</>}

type DocumentItem = {
  id: string;
  tenant_id: string;
  project_id: string;
  collection: string;
  external_document_id: string;
  current_version: number;
  lock_version: number;
  metadata: Record<string, unknown>;
};
type DocumentChunk = {id:string; chunk_index:number; chunk_type:string; content:string; token_count:number; language:string};
type ProjectOption = {id:string; tenant_id:string; name:string};
type CollectionOption = {id:string; project_id:string; name:string};

export function documentQuery(tenantId:string, projectId:string, collection:string) {
  const query = new URLSearchParams({tenant_id:tenantId, project_id:projectId, limit:"100"});
  if (collection) query.set("collection", collection);
  return query.toString();
}

export function Documents() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const projectId = params.get("project_id") ?? "";
  const collection = params.get("collection") ?? "";
  const documentId = params.get("document") ?? "";
  const projects = useQuery({queryKey:["projects"],queryFn:()=>api<ProjectOption[]>("/v1/admin/projects")});
  const collections = useQuery({queryKey:["collections"],queryFn:()=>api<CollectionOption[]>("/v1/admin/collections")});
  const selectedProject = projects.data?.find(item=>item.id===projectId);
  const documents = useQuery({
    queryKey:["documents",selectedProject?.tenant_id,projectId,collection],
    queryFn:()=>api<DocumentItem[]>(`/v1/admin/documents?${documentQuery(selectedProject?.tenant_id ?? "",projectId,collection)}`),
    enabled:Boolean(projectId && selectedProject),
  });
  const selectedDocument = documents.data?.find(item=>item.id===documentId);
  const chunks = useQuery({
    queryKey:["document-chunks",documentId],
    queryFn:()=>api<DocumentChunk[]>(`/v1/admin/documents/${documentId}/chunks?${new URLSearchParams({tenant_id:selectedDocument?.tenant_id ?? "",project_id:selectedDocument?.project_id ?? "",collection:selectedDocument?.collection ?? "",limit:"100"})}`),
    enabled:Boolean(documentId && selectedDocument),
  });
  const action = useMutation({
    mutationFn: ({document,action}:{document:DocumentItem; action:"reindex"|"delete"}) => {
      const scope=new URLSearchParams({tenant_id:document.tenant_id,project_id:document.project_id,collection:document.collection});
      return api(`/v1/admin/documents/${document.id}/${action}?${scope}`,{method:"POST",body:JSON.stringify({confirm:true})});
    },
    onSuccess:()=>client.invalidateQueries({queryKey:["documents"]}),
  });

  function filter(event:FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const next = new URLSearchParams({project_id:String(form.get("project_id") ?? "")});
    const selectedCollection = String(form.get("collection") ?? "");
    if (selectedCollection) next.set("collection",selectedCollection);
    setParams(next);
  }
  const availableCollections = collections.data?.filter(item=>item.project_id===projectId) ?? [];
  async function act(document:DocumentItem, operation:"reindex"|"delete") {
    if (!window.confirm(`${operation === "reindex" ? "Reindex" : "Delete"} document ${document.external_document_id}?`)) return;
    await action.mutateAsync({document,action:operation});
  }

  return <>
    <header><div><h1>Documents</h1><p className="lede">Tenant-scoped document inventory and indexed chunk inspection.</p></div></header>
    <form className="document-filter" onSubmit={filter}>
      <label>Project<select name="project_id" value={projectId} required onChange={event=>setParams(event.target.value?{project_id:event.target.value}:{})}><option value="">Select a project</option>{projects.data?.map(item=><option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
      <label>Collection<select name="collection" defaultValue={collection}><option value="">All authorized collections</option>{availableCollections.map(item=><option key={item.id} value={item.name}>{item.name}</option>)}</select></label>
      <button>Load documents</button>
    </form>
    {!projectId && <div className="empty">Select a project to load its authorized documents.</div>}
    {documents.isLoading && <Loading/>}{(documents.error||action.error) && <p role="alert">{(documents.error??action.error)?.message}</p>}
    {projectId && documents.data?.length===0 && <div className="empty">No documents match this scope.</div>}
    {!!documents.data?.length && <div className="split-view"><table><thead><tr><th>External ID</th><th>Collection</th><th>Version</th><th>Metadata</th><th>Actions</th></tr></thead><tbody>{documents.data.map(item=><tr key={item.id} className={item.id===documentId?"selected-row":undefined}>
      <td>{item.external_document_id}<small className="resource-id">{item.id}</small></td><td>{item.collection}</td><td>{item.current_version}</td><td>{Object.keys(item.metadata).length} fields</td>
      <td><div className="actions"><button className="quiet" onClick={()=>{const next=new URLSearchParams(params);next.set("document",item.id);setParams(next)}}>Chunks</button><button disabled={action.isPending} onClick={()=>act(item,"reindex")}>Reindex</button><button className="danger" disabled={action.isPending} onClick={()=>act(item,"delete")}>Delete</button></div></td>
    </tr>)}</tbody></table>
      {documentId && <aside className="detail-panel"><div className="setting-heading"><h2>Document chunks</h2><button className="quiet" onClick={()=>{const next=new URLSearchParams(params);next.delete("document");setParams(next)}}>Close</button></div>
        {chunks.isLoading&&<Loading/>}{chunks.error&&<p role="alert">{chunks.error.message}</p>}{chunks.data?.length===0&&<div className="empty">No chunks available.</div>}
        {chunks.data?.map(chunk=><article className="chunk-card" key={chunk.id}><div><span>Chunk {chunk.chunk_index}</span><small>{chunk.chunk_type} · {chunk.token_count} tokens · {chunk.language}</small></div><p>{chunk.content}</p></article>)}
      </aside>}
    </div>}
  </>;
}

type EvaluationDatasetItem = {id:string; tenant_id:string; project_id:string; name:string; version:number; collections:string[]; case_count:number};
type EvaluationRunItem = {id:string; tenant_id:string; project_id:string; dataset_id:string; status:string; configuration:Record<string,unknown>; metrics_before_reranking?:Record<string,number>; metrics_after_reranking?:Record<string,number>; reranker_uplift?:Record<string,number>; results?:Array<Record<string,unknown>>};
type EvaluationComparison = {baseline:{id:string;configuration:Record<string,unknown>};candidate:{id:string;configuration:Record<string,unknown>};comparison:Array<{metric:string;baseline:number;candidate:number;delta:number}>};

export const evaluationScope = (tenantId:string, projectId:string) =>
  new URLSearchParams({tenant_id:tenantId, project_id:projectId}).toString();

function MetricComparison({run}:{run:EvaluationRunItem}) {
  const before=run.metrics_before_reranking??{};
  const after=run.metrics_after_reranking??{};
  const uplift=run.reranker_uplift??{};
  const names=[...new Set([...Object.keys(before),...Object.keys(after)])].filter(name=>!name.includes("latency")&&!name.includes("violations")).sort();
  if (!names.length) return <div className="empty">Metrics are available after the run completes.</div>;
  return <table><thead><tr><th>Metric</th><th>Before</th><th>After</th><th>Uplift</th></tr></thead><tbody>{names.map(name=><tr key={name}><td>{name}</td><td>{(before[name]??0).toFixed(4)}</td><td>{(after[name]??0).toFixed(4)}</td><td>{(uplift[name]??0).toFixed(4)}</td></tr>)}</tbody></table>;
}

export function Evaluation() {
  const client = useQueryClient();
  const [params,setParams] = useSearchParams();
  const projectId = params.get("project_id") ?? "";
  const runId = params.get("run") ?? "";
  const [baselineRunId,setBaselineRunId] = useState("");
  const [candidateRunId,setCandidateRunId] = useState("");
  const projects = useQuery({queryKey:["projects"],queryFn:()=>api<ProjectOption[]>("/v1/admin/projects")});
  const project = projects.data?.find(item=>item.id===projectId);
  const scope = project ? evaluationScope(project.tenant_id,project.id) : "";
  const datasets = useQuery({queryKey:["evaluation-datasets",scope],queryFn:()=>api<EvaluationDatasetItem[]>(`/v1/admin/evaluation/datasets?${scope}`),enabled:Boolean(scope)});
  const runs = useQuery({queryKey:["evaluation-runs",scope],queryFn:()=>api<EvaluationRunItem[]>(`/v1/admin/evaluation/runs?${scope}`),enabled:Boolean(scope),refetchInterval:10000});
  const detail = useQuery({queryKey:["evaluation-run",runId,scope],queryFn:()=>api<EvaluationRunItem>(`/v1/admin/evaluation/runs/${runId}?${scope}`),enabled:Boolean(runId&&scope)});
  const start = useMutation({
    mutationFn:(datasetId:string)=>api(`/v1/admin/evaluation/runs?${scope}`,{method:"POST",body:JSON.stringify({dataset_id:datasetId})}),
    onSuccess:()=>client.invalidateQueries({queryKey:["evaluation-runs"]}),
  });
  const compare = useMutation({
    mutationFn:()=>api<EvaluationComparison>(`/v1/admin/evaluation/compare?${scope}`,{method:"POST",body:JSON.stringify({baseline_run_id:baselineRunId,candidate_run_id:candidateRunId})}),
  });
  async function run(dataset:EvaluationDatasetItem) {
    if (!window.confirm(`Run evaluation dataset ${dataset.name} v${dataset.version}?`)) return;
    await start.mutateAsync(dataset.id);
  }
  function chooseProject(value:string) {setParams(value?{project_id:value}:{})}

  return <>
    <header><div><h1>Evaluation</h1><p className="lede">Versioned datasets, controlled runs, and retrieval quality metrics.</p></div>
      <label>Project<select value={projectId} onChange={event=>chooseProject(event.target.value)}><option value="">Select a project</option>{projects.data?.map(item=><option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
    </header>
    {!projectId&&<div className="empty">Select a project to inspect evaluation datasets and runs.</div>}
    {(datasets.isLoading||runs.isLoading)&&<Loading/>}{(datasets.error||runs.error||detail.error||start.error||compare.error)&&<p role="alert">{(datasets.error??runs.error??detail.error??start.error??compare.error)?.message}</p>}
    {projectId&&<div className="evaluation-grid"><section><h2>Datasets</h2>{datasets.data?.length===0&&<div className="empty">No evaluation datasets.</div>}{datasets.data?.map(dataset=><article className="evaluation-item" key={dataset.id}>
      <div><h3>{dataset.name} <span className="badge">v{dataset.version}</span></h3><small>{dataset.collections.join(", ")} · {dataset.case_count} cases</small></div>
      <button disabled={start.isPending} onClick={()=>run(dataset)}>Run evaluation</button>
    </article>)}</section><section><h2>Runs</h2>{runs.data?.length===0&&<div className="empty">No evaluation runs.</div>}{runs.data?.map(item=><article className="evaluation-item" key={item.id}>
      <div><span className={`badge status-${item.status}`}>{item.status}</span><small className="resource-id">{item.id}</small></div>
      <button className="quiet" onClick={()=>{const next=new URLSearchParams(params);next.set("run",item.id);setParams(next)}}>Results</button>
    </article>)}</section></div>}
    {projectId&&<section className="result-panel"><h2>Compare completed runs</h2><form className="comparison-controls" onSubmit={event=>{event.preventDefault();compare.mutate()}}>
      <label>Baseline<select value={baselineRunId} onChange={event=>setBaselineRunId(event.target.value)} required><option value="">Select baseline</option>{runs.data?.filter(item=>item.status==="completed").map(item=><option key={item.id} value={item.id}>{item.id}</option>)}</select></label>
      <label>Candidate<select value={candidateRunId} onChange={event=>setCandidateRunId(event.target.value)} required><option value="">Select candidate</option>{runs.data?.filter(item=>item.status==="completed").map(item=><option key={item.id} value={item.id}>{item.id}</option>)}</select></label>
      <button disabled={compare.isPending||!baselineRunId||!candidateRunId}>Compare</button>
    </form>{compare.data&&<><table><thead><tr><th>Metric</th><th>Baseline</th><th>Candidate</th><th>Delta</th></tr></thead><tbody>{compare.data.comparison.map(row=><tr key={row.metric}><td>{row.metric}</td><td>{row.baseline.toFixed(4)}</td><td>{row.candidate.toFixed(4)}</td><td className={row.delta<0?"metric-regression":"metric-improvement"}>{row.delta>=0?"+":""}{row.delta.toFixed(4)}</td></tr>)}</tbody></table><details><summary>Configurations</summary><pre>{JSON.stringify({baseline:compare.data.baseline,candidate:compare.data.candidate},null,2)}</pre></details></>}
    </section>}
    {runId&&detail.data&&<aside className="result-panel"><div className="setting-heading"><div><h2>Run results</h2><small className="mono">{detail.data.id}</small></div><button className="quiet" onClick={()=>{const next=new URLSearchParams(params);next.delete("run");setParams(next)}}>Close</button></div>
      <dl><dt>Status</dt><dd>{detail.data.status}</dd><dt>Dataset</dt><dd className="mono">{detail.data.dataset_id}</dd><dt>Result cases</dt><dd>{detail.data.results?.length??0}</dd></dl>
      <MetricComparison run={detail.data}/>
      <details><summary>Case-level results</summary><pre>{JSON.stringify(detail.data.results??[],null,2)}</pre></details>
    </aside>}
  </>;
}

type FeedbackItem = {id:string; created_at:string; request_id:string; chunk_id:string; collection:string; relevant:boolean; relevance_grade:number|null; comment:string|null};

export function feedbackScope(tenantId:string, projectId:string, relevant:string, collection:string) {
  const query = new URLSearchParams({tenant_id:tenantId,project_id:projectId,limit:"100"});
  if (relevant) query.set("relevant",relevant);
  if (collection) query.set("collection",collection);
  return query.toString();
}

export function Feedback() {
  const [params,setParams] = useSearchParams();
  const projectId = params.get("project_id") ?? "";
  const relevant = params.get("relevant") ?? "";
  const collection = params.get("collection") ?? "";
  const projects = useQuery({queryKey:["projects"],queryFn:()=>api<ProjectOption[]>("/v1/admin/projects")});
  const project = projects.data?.find(item=>item.id===projectId);
  const scope = project ? feedbackScope(project.tenant_id,project.id,relevant,collection) : "";
  const query = useQuery({queryKey:["feedback",scope],queryFn:()=>api<FeedbackItem[]>(`/v1/admin/feedback?${scope}`),enabled:Boolean(scope)});
  function filter(event:FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const next = new URLSearchParams();
    for (const key of ["project_id","relevant","collection"]) {
      const value = String(form.get(key)??"").trim();
      if (value) next.set(key,value);
    }
    setParams(next);
  }
  const positive = query.data?.filter(item=>item.relevant).length ?? 0;

  return <>
    <header><div><h1>Feedback</h1><p className="lede">Retrieval relevance judgments scoped to an explicit tenant and project.</p></div></header>
    <form className="feedback-filter" onSubmit={filter}>
      <label>Project<select name="project_id" defaultValue={projectId} required><option value="">Select a project</option>{projects.data?.map(item=><option key={item.id} value={item.id}>{item.name}</option>)}</select></label>
      <label>Judgment<select name="relevant" defaultValue={relevant}><option value="">All judgments</option><option value="true">Relevant</option><option value="false">Not relevant</option></select></label>
      <label>Collection<input name="collection" defaultValue={collection} placeholder="manuals"/></label><button>Apply filters</button>
    </form>
    {!projectId&&<div className="empty">Select a project to inspect retrieval feedback.</div>}
    {query.isLoading&&<Loading/>}{query.error&&<p role="alert">{query.error.message}</p>}
    {!!query.data&&<section className="cards feedback-cards"><article><span>Judgments</span><strong>{query.data.length}</strong></article><article><span>Relevant</span><strong className="healthy">{positive}</strong></article><article><span>Not relevant</span><strong className="unhealthy">{query.data.length-positive}</strong></article></section>}
    {query.data?.length===0&&<div className="empty">No feedback matches this scope.</div>}
    {!!query.data?.length&&<table><thead><tr><th>Time</th><th>Judgment</th><th>Grade</th><th>Collection</th><th>Request / chunk</th><th>Comment</th></tr></thead><tbody>{query.data.map(item=><tr key={item.id}>
      <td>{new Date(item.created_at).toLocaleString()}</td><td><span className={`badge ${item.relevant?"status-completed":"status-failed"}`}>{item.relevant?"Relevant":"Not relevant"}</span></td><td>{item.relevance_grade??"—"}</td><td>{item.collection}</td>
      <td><small className="mono">{item.request_id}</small><small className="mono resource-id">{item.chunk_id}</small></td><td>{item.comment??"—"}</td>
    </tr>)}</tbody></table>}
  </>;
}
type PlaygroundResult = {
  request_id:string;
  results:Array<Record<string,any>>;
  trace:Record<string,any>;
};

function RankingTable({title,items,score}:{title:string;items:Array<Record<string,any>>;score:string}) {
  return <section className="result-panel"><h2>{title}</h2>{items.length===0?<p className="empty">No candidates.</p>:<table><thead><tr><th>Rank</th><th>Document / chunk</th><th>Score</th><th>Preview</th></tr></thead><tbody>{items.map((item,index)=><tr key={`${title}-${item.chunk_id}`}><td>{item.final_rank??index+1}</td><td><span className="mono">{item.document_id}</span><br/><small className="mono">{item.chunk_id}</small></td><td>{typeof item[score]==="number"?item[score].toFixed(5):"—"}</td><td>{String(item.content??"").slice(0,180)}</td></tr>)}</tbody></table>}</section>
}

export function SearchPlayground(){
  const [result,setResult]=useState<PlaygroundResult>();
  const [error,setError]=useState<string>();
  async function run(e:FormEvent<HTMLFormElement>){
    e.preventDefault();
    setError(undefined);
    const f=new FormData(e.currentTarget);
    const topK=Number(f.get("top_k")??20);
    try {
      setResult(await api<PlaygroundResult>("/v1/admin/retrieval/search",{method:"POST",body:JSON.stringify({
        project_id:f.get("project_id"),
        collections:String(f.get("collections")).split(",").map(value=>value.trim()).filter(Boolean),
        query:f.get("query"),
        mode:f.get("mode"),
        vector_top_k:topK,
        bm25_top_k:topK,
        fusion_top_k:topK,
        use_reranker:f.get("use_reranker")==="on",
        rerank_top_k:Number(f.get("rerank_top_k")??8),
        include_trace:true,
      })}));
    } catch (value) { setError(value instanceof Error?value.message:"Search failed"); }
  }
  const results=result?.results??[];
  const dense=[...results].filter(item=>typeof item.vector_score==="number").sort((a,b)=>b.vector_score-a.vector_score);
  const lexical=[...results].filter(item=>typeof item.bm25_score==="number").sort((a,b)=>b.bm25_score-a.bm25_score);
  const fusion=[...results].sort((a,b)=>(b.fusion_score??0)-(a.fusion_score??0));
  const reranked=[...results].filter(item=>typeof item.reranker_score==="number").sort((a,b)=>b.reranker_score-a.reranker_score);
  return <><header><div><h1>Search Playground</h1><p className="lede">Compare retrieval stages and degraded fallbacks with one scoped query.</p></div></header><form onSubmit={run} className="filters"><input name="project_id" placeholder="Project UUID" required/><input name="collections" placeholder="Collection names" required/><select name="mode" aria-label="Retrieval mode" defaultValue="hybrid"><option value="lexical">Lexical</option><option value="dense">Dense</option><option value="hybrid">Hybrid</option></select><label>Candidate top K<input name="top_k" type="number" min="1" max="200" defaultValue="20"/></label><label>Final top K<input name="rerank_top_k" type="number" min="1" max="50" defaultValue="8"/></label><label><input name="use_reranker" type="checkbox" defaultChecked/> External reranker</label><textarea name="query" placeholder="Ask a retrieval question" required/><button>Search</button></form>{error&&<p role="alert">{error}</p>}{result&&<><section className="cards"><article><span>Requested mode</span><strong>{result.trace.requested_mode}</strong></article><article><span>Effective mode</span><strong>{result.trace.effective_mode}</strong></article><article><span>Total latency</span><strong>{result.trace.latency_ms} ms</strong></article><article><span>Reranker</span><strong>{result.trace.reranker_degraded?"Degraded":result.trace.reranker_used?"Used":"Skipped"}</strong></article></section><RankingTable title="Dense ranking" items={dense} score="vector_score"/><RankingTable title="Lexical ranking" items={lexical} score="bm25_score"/><RankingTable title="Fusion ranking" items={fusion} score="fusion_score"/><RankingTable title="Reranker ranking" items={reranked} score="reranker_score"/><RankingTable title="Final selected results" items={results} score="final_score"/><details><summary>Trace and stage timings</summary><pre>{JSON.stringify(result.trace,null,2)}</pre></details></>}</>;
}
export function SystemHealth(){const q=useQuery({queryKey:["health"],queryFn:()=>api<any>("/v1/admin/system/health"),refetchInterval:10000});return <><h1>System Health</h1>{q.isLoading?<Loading/>:<section className="cards">{q.data?.components.map((c:any)=><article key={c.name}><span>{c.name}</span><strong className="healthy">{c.status}</strong></article>)}</section>}</>}

type SettingItem = {
  key: string;
  value: boolean | number;
  default: boolean | number;
  description: string;
  minimum: number | null;
  maximum: number | null;
  restart_required: boolean;
  reindex_required: boolean;
};

type SettingsResponse = {settings: SettingItem[]};

export function settingValue(raw: FormDataEntryValue, current: boolean | number) {
  return typeof current === "boolean" ? raw === "true" : Number(raw);
}

export function Settings() {
  const client = useQueryClient();
  const query = useQuery({
    queryKey: ["settings"],
    queryFn: () => api<SettingsResponse>("/v1/admin/settings"),
  });
  const mutation = useMutation({
    mutationFn: (update: Record<string, boolean | number>) =>
      api<SettingsResponse>("/v1/admin/settings", {
        method: "PATCH",
        body: JSON.stringify(update),
      }),
    onSuccess: (data) => client.setQueryData(["settings"], data),
  });
  const cacheClear = useMutation({
    mutationFn: () => api<{status: string; deleted_keys: number}>("/v1/admin/cache/clear", {
      method: "POST",
      body: JSON.stringify({confirm: true}),
    }),
  });

  if (query.isLoading) return <Loading />;
  if (query.error) return <p role="alert">{query.error.message}</p>;

  async function save(event: FormEvent<HTMLFormElement>, setting: SettingItem) {
    event.preventDefault();
    const raw = new FormData(event.currentTarget).get("value");
    if (raw === null) return;
    await mutation.mutateAsync({[setting.key]: settingValue(raw, setting.value)});
  }

  function clearCache() {
    if (window.confirm("Clear only the versioned RAG cache namespace?")) cacheClear.mutate();
  }

  return <>
    <header><div><h1>Settings</h1><p className="lede">Validated runtime controls. Secrets are never displayed.</p></div></header>
    {mutation.error && <p role="alert">{mutation.error.message}</p>}
    <section className="settings-grid">
      {query.data?.settings.map(setting => <article className="setting" key={setting.key}>
        <div className="setting-heading">
          <div><strong>{setting.key.replaceAll("_", " ")}</strong><p>{setting.description}</p></div>
          <div className="badges">
            {setting.restart_required && <span className="badge warning">Restart</span>}
            {setting.reindex_required && <span className="badge warning">Reindex</span>}
          </div>
        </div>
        <form onSubmit={event => save(event, setting)}>
          <label>Current value
            {typeof setting.value === "boolean"
              ? <select name="value" defaultValue={String(setting.value)}><option value="true">Enabled</option><option value="false">Disabled</option></select>
              : <input name="value" type="number" required defaultValue={setting.value} min={setting.minimum ?? undefined} max={setting.maximum ?? undefined}/>
            }
          </label>
          <div className="setting-footer"><small>Default: {String(setting.default)}</small><button disabled={mutation.isPending}>Save</button></div>
        </form>
      </article>)}
    </section>
    <article className="setting">
      <div className="setting-heading"><div><strong>RAG cache</strong><p>Query embeddings are versioned by model identity. Clearing does not flush Celery or unrelated Redis keys.</p></div></div>
      <div className="setting-footer"><small>Use after an emergency cache investigation; normal model changes are isolated automatically.</small><button className="danger" disabled={cacheClear.isPending} onClick={clearCache}>Clear RAG cache</button></div>
      {cacheClear.data && <p role="status">Cleared {cacheClear.data.deleted_keys} cache keys.</p>}
      {cacheClear.error && <p role="alert">{cacheClear.error.message}</p>}
    </article>
  </>;
}

type AuditEvent = {
  id: string;
  tenant_id: string;
  project_id: string | null;
  created_at: string;
  action: string;
  resource_type: string;
  resource_id: string;
};

export function auditQuery(params: URLSearchParams) {
  const query = new URLSearchParams();
  for (const key of ["action", "tenant_id", "project_id", "limit"]) {
    const value = params.get(key)?.trim();
    if (value) query.set(key, value);
  }
  if (!query.has("limit")) query.set("limit", "100");
  return query.toString();
}

export function AuditLog() {
  const [params, setParams] = useSearchParams();
  const queryString = auditQuery(params);
  const query = useQuery({
    queryKey: ["audit-log", queryString],
    queryFn: () => api<AuditEvent[]>(`/v1/admin/audit-log?${queryString}`),
  });

  function filter(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = new FormData(event.currentTarget);
    const next = new URLSearchParams();
    for (const key of ["action", "tenant_id", "project_id", "limit"]) {
      const value = String(form.get(key) ?? "").trim();
      if (value) next.set(key, value);
    }
    setParams(next);
  }

  return <>
    <header><div><h1>Audit Log</h1><p className="lede">Administrative mutations with safe resource metadata.</p></div></header>
    <form className="filters" onSubmit={filter}>
      <label>Action<input name="action" defaultValue={params.get("action") ?? ""} placeholder="project.update"/></label>
      <label>Tenant UUID<input name="tenant_id" defaultValue={params.get("tenant_id") ?? ""}/></label>
      <label>Project UUID<input name="project_id" defaultValue={params.get("project_id") ?? ""}/></label>
      <label>Limit<input name="limit" type="number" min="1" max="500" defaultValue={params.get("limit") ?? "100"}/></label>
      <button>Apply filters</button>
    </form>
    {query.isLoading && <Loading/>}
    {query.error && <p role="alert">{query.error.message}</p>}
    {query.data?.length === 0 && <div className="empty">No audit events match the current filters.</div>}
    {!!query.data?.length && <table>
      <thead><tr><th>Time</th><th>Action</th><th>Resource</th><th>Tenant</th><th>Project</th></tr></thead>
      <tbody>{query.data.map(event => <tr key={event.id}>
        <td>{new Date(event.created_at).toLocaleString()}</td>
        <td><span className="badge">{event.action}</span></td>
        <td>{event.resource_type}<small className="resource-id">{event.resource_id}</small></td>
        <td className="mono">{event.tenant_id}</td>
        <td className="mono">{event.project_id ?? "Global"}</td>
      </tr>)}</tbody>
    </table>}
  </>;
}

type IndexingJob = {
  id: string;
  tenant_id: string;
  project_id: string | null;
  status: string;
  stage?: string;
  job_type?: string;
  error?: string | null;
};

export const canRetryJob = (status: string) => status === "failed" || status === "dead_letter";
export const canCancelJob = (status: string) => status === "queued";

export function Indexing() {
  const client = useQueryClient();
  const [params, setParams] = useSearchParams();
  const status = params.get("status") ?? "";
  const queryString = new URLSearchParams({limit: "100"});
  if (status) queryString.set("status", status);
  const query = useQuery({
    queryKey: ["indexing-jobs", queryString.toString()],
    queryFn: () => api<IndexingJob[]>(`/v1/admin/indexing/jobs?${queryString}`),
    refetchInterval: 10000,
  });
  const profileQuery = useQuery({
    queryKey: ["embedding-profile"],
    queryFn: () => api<EmbeddingProfile>("/v1/admin/models/embeddings"),
    refetchInterval: 10000,
  });
  const mutation = useMutation({
    mutationFn: ({id, action}:{id:string; action:"retry"|"cancel"}) =>
      api(`/v1/admin/indexing/jobs/${id}/${action}`, {method: "POST"}),
    onSuccess: () => client.invalidateQueries({queryKey: ["indexing-jobs"]}),
  });

  async function act(job: IndexingJob, action: "retry" | "cancel") {
    if (!window.confirm(`${action === "retry" ? "Retry" : "Cancel"} job ${job.id}?`)) return;
    await mutation.mutateAsync({id: job.id, action});
  }

  return <>
    <header><div><h1>Indexing</h1><p className="lede">Queue state and controlled recovery actions.</p></div>
      <label>Status<select value={status} onChange={event => setParams(event.target.value ? {status:event.target.value} : {})}>
        <option value="">All statuses</option><option value="queued">Queued</option><option value="processing">Processing</option><option value="failed">Failed</option><option value="dead_letter">Dead letter</option><option value="completed">Completed</option>
      </select></label>
    </header>
    {profileQuery.data && <section className="cards"><article><span>Active index</span><strong>{profileQuery.data.index_version}</strong></article><article><span>Embedding model</span><strong>{profileQuery.data.model}</strong><small>{profileQuery.data.revision}</small></article><article><span>Dimension</span><strong>{profileQuery.data.dimension ?? profileQuery.data.expected_dimension}</strong></article><article><span>Chunker</span><strong>{profileQuery.data.chunker_version}</strong></article></section>}
    {mutation.error && <p role="alert">{mutation.error.message}</p>}
    {query.isLoading && <Loading/>}
    {query.error && <p role="alert">{query.error.message}</p>}
    {query.data?.length === 0 && <div className="empty">No indexing jobs match this status.</div>}
    {!!query.data?.length && <table><thead><tr><th>Status</th><th>Type / stage</th><th>Project</th><th>Error</th><th>Actions</th></tr></thead>
      <tbody>{query.data.map(job => <tr key={job.id}>
        <td><span className={`badge status-${job.status}`}>{job.status}</span></td>
        <td>{job.job_type ?? "document.index"}<small className="resource-id">{job.stage ?? "Pending"}</small></td>
        <td className="mono">{job.project_id ?? "Global"}</td><td>{job.error ?? "—"}</td>
        <td><div className="actions">
          {canRetryJob(job.status) && <button disabled={mutation.isPending} onClick={() => act(job,"retry")}>Retry</button>}
          {canCancelJob(job.status) && <button className="danger" disabled={mutation.isPending} onClick={() => act(job,"cancel")}>Cancel</button>}
          {!canRetryJob(job.status) && !canCancelJob(job.status) && <small>No actions</small>}
        </div></td>
      </tr>)}</tbody></table>}
  </>;
}

type RetrievalTrace = {
  id: string;
  project_id: string;
  created_at: string;
  query: string;
  collections: string[];
  configuration: Record<string, unknown>;
  results: Array<Record<string, unknown>>;
  trace: {latency_ms?: number; reranker_degraded?: boolean; opensearch_degraded?: boolean};
};

export const traceStatus = (trace: RetrievalTrace) =>
  trace.trace.reranker_degraded || trace.trace.opensearch_degraded ? "Degraded" : "Healthy";

export function RetrievalTraces() {
  const [params, setParams] = useSearchParams();
  const selectedId = params.get("trace");
  const query = useQuery({
    queryKey: ["retrieval-traces"],
    queryFn: () => api<RetrievalTrace[]>("/v1/admin/retrieval/traces?limit=100"),
  });
  const mutation = useMutation({
    mutationFn: (id:string) => api(`/v1/admin/retrieval/traces/${id}/repeat`, {method:"POST"}),
  });
  const selected = query.data?.find(trace => trace.id === selectedId);

  async function repeat(trace: RetrievalTrace) {
    if (!window.confirm(`Repeat retrieval query "${trace.query}"?`)) return;
    await mutation.mutateAsync(trace.id);
  }

  return <>
    <header><div><h1>Retrieval Traces</h1><p className="lede">Inspect stored retrieval decisions and degraded fallbacks.</p></div></header>
    {query.isLoading && <Loading/>}{query.error && <p role="alert">{query.error.message}</p>}
    {mutation.error && <p role="alert">{mutation.error.message}</p>}
    {query.data?.length === 0 && <div className="empty">No retrieval traces have been recorded.</div>}
    {!!query.data?.length && <div className="split-view"><table><thead><tr><th>Time</th><th>Query</th><th>Collections</th><th>Latency</th><th>Status</th><th></th></tr></thead>
      <tbody>{query.data.map(trace => <tr key={trace.id} className={trace.id===selectedId?"selected-row":undefined}>
        <td>{new Date(trace.created_at).toLocaleString()}</td><td>{trace.query}</td><td>{trace.collections.join(", ")}</td>
        <td>{trace.trace.latency_ms ?? "—"} ms</td><td><span className={`badge ${traceStatus(trace)==="Degraded"?"warning":"status-completed"}`}>{traceStatus(trace)}</span></td>
        <td><button className="quiet" onClick={() => setParams({trace:trace.id})}>Inspect</button></td>
      </tr>)}</tbody></table>
      {selected && <aside className="detail-panel"><div className="setting-heading"><h2>Trace detail</h2><button className="quiet" onClick={() => setParams({})}>Close</button></div>
        <dl><dt>Request ID</dt><dd className="mono">{selected.id}</dd><dt>Project</dt><dd className="mono">{selected.project_id}</dd><dt>Results</dt><dd>{selected.results.length}</dd></dl>
        <h3>Configuration</h3><pre>{JSON.stringify(selected.configuration,null,2)}</pre><h3>Trace</h3><pre>{JSON.stringify(selected.trace,null,2)}</pre>
        <button disabled={mutation.isPending} onClick={() => repeat(selected)}>Repeat query</button>
      </aside>}
    </div>}
  </>;
}

type EmbeddingProfile = {
  status: string;
  model: string;
  backend: string;
  revision: string;
  normalization: string;
  device: string | null;
  dimension: number | null;
  expected_dimension: number;
  chunker_version: string;
  index_version: string;
  compatible: boolean;
};

export const embeddingCompatibility = (profile: EmbeddingProfile) =>
  profile.compatible ? "Compatible" : "Incompatible";

export function Models() {
  const query = useQuery({
    queryKey: ["embedding-profile"],
    queryFn: () => api<EmbeddingProfile>("/v1/admin/models/embeddings"),
    refetchInterval: 10000,
  });
  const check = useMutation({mutationFn: () => api<EmbeddingProfile>("/v1/admin/models/embeddings/check", {method:"POST"})});
  const reindex = useMutation({mutationFn: () => api<Record<string,number>>("/v1/admin/models/embeddings/reindex", {method:"POST",body:JSON.stringify({confirm:true})})});
  const profile = check.data ?? query.data;

  async function startReindex() {
    if (!profile?.compatible) return;
    if (!window.confirm(`Reindex every current document with ${profile.model}?`)) return;
    await reindex.mutateAsync();
  }

  return <>
    <header><div><h1>Models</h1><p className="lede">Embedding worker compatibility and guarded reindexing.</p></div>
      <button className="quiet" disabled={check.isPending} onClick={() => check.mutate()}>Check compatibility</button>
    </header>
    {query.isLoading && <Loading/>}{query.error && <p role="alert">{query.error.message}</p>}
    {(check.error || reindex.error) && <p role="alert">{(check.error ?? reindex.error)?.message}</p>}
    {profile && <><section className="cards model-cards">
      <article><span>Status</span><strong className={profile.status==="ready"?"healthy":""}>{profile.status}</strong></article>
      <article><span>Compatibility</span><strong className={profile.compatible?"healthy":"unhealthy"}>{embeddingCompatibility(profile)}</strong></article>
      <article><span>Device</span><strong>{profile.device ?? "Unknown"}</strong></article>
      <article><span>Dimension</span><strong>{profile.dimension ?? "—"} / {profile.expected_dimension}</strong></article>
      <article><span>Index</span><strong>{profile.index_version}</strong><small>{profile.revision}</small></article>
    </section>
    <article className="model-profile"><div><span>Active embedding model</span><h2>{profile.model}</h2></div>
      <p>Reindexing preserves active jobs and requires a compatible worker heartbeat.</p>
      <button className="danger" disabled={!profile.compatible || reindex.isPending} onClick={startReindex}>Reindex all embeddings</button>
      {reindex.data && <p role="status">Requeued {reindex.data.requeued}; skipped active {reindex.data.skipped_active}.</p>}
    </article></>}
  </>;
}

type RerankerState = {
  status: "up" | "disabled" | "unavailable";
  latency_ms?: number;
  model?: string;
  version?: string;
  device?: string;
  error?: string;
  result_count?: number;
};

export const rerankerStatusLabel = (status: RerankerState["status"]) =>
  status === "up" ? "Available" : status === "disabled" ? "Disabled" : "Unavailable";

export function Reranker() {
  const query = useQuery({
    queryKey: ["reranker-status"],
    queryFn: () => api<RerankerState>("/v1/admin/reranker/status"),
    refetchInterval: 10000,
  });
  const test = useMutation({
    mutationFn: () => api<RerankerState>("/v1/admin/reranker/test", {method: "POST"}),
  });
  const state = test.data ?? query.data;

  return <>
    <header><div><h1>Reranker</h1><p className="lede">Connectivity and runtime identity without exposing service credentials.</p></div>
      <button className="quiet" disabled={test.isPending} onClick={() => test.mutate()}>Test connection</button>
    </header>
    {query.isLoading && <Loading/>}{query.error && <p role="alert">{query.error.message}</p>}
    {test.error && <p role="alert">{test.error.message}</p>}
    {state && <>
      {state.status !== "up" && <p className="degraded" role="status">Reranking is {state.status}. Retrieval remains available through the configured fallback.</p>}
      <section className="cards model-cards">
        <article><span>Status</span><strong className={state.status === "up" ? "healthy" : "unhealthy"}>{rerankerStatusLabel(state.status)}</strong></article>
        <article><span>Model</span><strong>{state.model ?? "Not reported"}</strong>{state.version && <small>Version {state.version}</small>}</article>
        <article><span>Device</span><strong>{state.device ?? "Not reported"}</strong></article>
        <article><span>Latency</span><strong>{state.latency_ms == null ? "—" : `${state.latency_ms} ms`}</strong></article>
      </section>
      {test.data && <article className="test-result"><div><span>Connection test</span><h2>{rerankerStatusLabel(test.data.status)}</h2></div>
        <p>{test.data.status === "up" ? `Returned ${test.data.result_count ?? 0} ranked results.` : `Safe error: ${test.data.error ?? test.data.status}.`}</p>
      </article>}
    </>}
  </>;
}
