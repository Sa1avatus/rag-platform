import {FormEvent, useState} from "react";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";
import {useSearchParams} from "react-router-dom";
import {api} from "./api";
const Loading=()=> <div className="skeleton">Loading…</div>;
export function Dashboard(){const q=useQuery({queryKey:["dashboard"],queryFn:()=>api<Record<string,number>>("/v1/admin/dashboard")});if(q.isLoading)return <Loading/>;if(q.error)return <p role="alert">{q.error.message}</p>;return <><header><h1>Dashboard</h1><select aria-label="Period"><option>Last 24 hours</option><option>7 days</option><option>30 days</option></select></header><section className="cards">{Object.entries(q.data??{}).map(([k,v])=><article key={k}><span>{k}</span><strong>{v.toLocaleString()}</strong></article>)}</section></>}
export function Projects(){const client=useQueryClient();const q=useQuery({queryKey:["projects"],queryFn:()=>api<any[]>("/v1/admin/projects")});async function create(e:FormEvent<HTMLFormElement>){e.preventDefault();const f=new FormData(e.currentTarget);await api("/v1/admin/projects",{method:"POST",body:JSON.stringify(Object.fromEntries(f))});client.invalidateQueries({queryKey:["projects"]});e.currentTarget.reset()}return <><h1>Projects</h1><form className="inline" onSubmit={create}><input name="tenant_id" placeholder="Tenant UUID" required/><input name="slug" placeholder="project-slug" required/><input name="name" placeholder="Project name" required/><button>Create</button></form>{q.isLoading?<Loading/>:<table><thead><tr><th>Name</th><th>Slug</th><th>Status</th></tr></thead><tbody>{q.data?.map(p=><tr key={p.id}><td>{p.name}</td><td>{p.slug}</td><td>{p.enabled?"Enabled":"Disabled"}</td></tr>)}</tbody></table>}</>}
export function Collections(){const q=useQuery({queryKey:["collections"],queryFn:()=>api<any[]>("/v1/admin/collections")});return <><h1>Collections</h1>{q.isLoading?<Loading/>:<table><thead><tr><th>Name</th><th>Project</th><th>Strategy</th></tr></thead><tbody>{q.data?.map(c=><tr key={c.id}><td>{c.name}</td><td>{c.project_id}</td><td>{c.settings.chunking_strategy??"recursive_text"}</td></tr>)}</tbody></table>}</>}
export function SearchPlayground(){const [result,setResult]=useState<any>();async function run(e:FormEvent<HTMLFormElement>){e.preventDefault();const f=new FormData(e.currentTarget);setResult(await api("/v1/admin/retrieval/search",{method:"POST",body:JSON.stringify({project_id:f.get("project_id"),collections:String(f.get("collections")).split(","),query:f.get("query"),include_trace:true})}))}return <><h1>Search Playground</h1><form onSubmit={run}><input name="project_id" placeholder="Project UUID" required/><input name="collections" placeholder="Collection names" required/><textarea name="query" placeholder="Ask a retrieval question" required/><button>Search</button></form>{result&&<pre>{JSON.stringify(result,null,2)}</pre>}</>}
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

  if (query.isLoading) return <Loading />;
  if (query.error) return <p role="alert">{query.error.message}</p>;

  async function save(event: FormEvent<HTMLFormElement>, setting: SettingItem) {
    event.preventDefault();
    const raw = new FormData(event.currentTarget).get("value");
    if (raw === null) return;
    await mutation.mutateAsync({[setting.key]: settingValue(raw, setting.value)});
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
  device: string | null;
  dimension: number | null;
  expected_dimension: number;
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
export function Placeholder({title}:{title:string}){return <><h1>{title}</h1><div className="empty">No records match the current filters.</div></>}
