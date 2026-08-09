import {FormEvent, useState} from "react";
import {useMutation, useQuery, useQueryClient} from "@tanstack/react-query";
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
export function Placeholder({title}:{title:string}){return <><h1>{title}</h1><div className="empty">No records match the current filters.</div></>}
