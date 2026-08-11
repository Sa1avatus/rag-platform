import {describe, expect, it} from "vitest";
import {auditQuery, canCancelJob, canRetryJob, documentQuery, embeddingCompatibility, evaluationScope, feedbackScope, rerankerStatusLabel, settingValue, traceStatus} from "./pages";

describe("settingValue", () => {
  it("parses boolean controls", () => {
    expect(settingValue("false", true)).toBe(false);
    expect(settingValue("true", false)).toBe(true);
  });

  it("parses numeric controls", () => {
    expect(settingValue("42", 30)).toBe(42);
  });
});

describe("auditQuery", () => {
  it("keeps supported URL filters and supplies a limit", () => {
    const params = new URLSearchParams("action=project.update&ignored=value");
    expect(auditQuery(params)).toBe("action=project.update&limit=100");
  });
});

describe("indexing job actions", () => {
  it("only enables actions for safe states", () => {
    expect(canRetryJob("failed")).toBe(true);
    expect(canRetryJob("processing")).toBe(false);
    expect(canCancelJob("queued")).toBe(true);
    expect(canCancelJob("processing")).toBe(false);
  });
});

describe("traceStatus", () => {
  it("surfaces degraded retrieval dependencies", () => {
    const base = {id:"1",project_id:"2",created_at:"",query:"q",collections:[],configuration:{},results:[]};
    expect(traceStatus({...base,trace:{reranker_degraded:true}})).toBe("Degraded");
    expect(traceStatus({...base,trace:{opensearch_degraded:false}})).toBe("Healthy");
  });
});

describe("embeddingCompatibility", () => {
  it("labels the verified model profile", () => {
    const profile = {status:"ready",model:"BAAI/bge-m3",backend:"sentence-transformers",revision:"default",normalization:"l2",device:"cpu",dimension:1024,expected_dimension:1024,chunker_version:"word-window-v1",index_version:"rag-chunks-v1",compatible:true};
    expect(embeddingCompatibility(profile)).toBe("Compatible");
    expect(embeddingCompatibility({...profile,compatible:false})).toBe("Incompatible");
  });
});

describe("rerankerStatusLabel", () => {
  it("surfaces healthy and degraded service states", () => {
    expect(rerankerStatusLabel("up")).toBe("Available");
    expect(rerankerStatusLabel("disabled")).toBe("Disabled");
    expect(rerankerStatusLabel("unavailable")).toBe("Unavailable");
  });
});

describe("documentQuery", () => {
  it("keeps the required project scope and optional collection", () => {
    expect(documentQuery("tenant-id", "project-id", "knowledge")).toBe("tenant_id=tenant-id&project_id=project-id&limit=100&collection=knowledge");
    expect(documentQuery("tenant-id", "project-id", "")).toBe("tenant_id=tenant-id&project_id=project-id&limit=100");
  });
});

describe("evaluationScope", () => {
  it("requires both tenant and project", () => {
    expect(evaluationScope("tenant-id","project-id")).toBe("tenant_id=tenant-id&project_id=project-id");
  });
});

describe("feedbackScope", () => {
  it("preserves scope and optional filters", () => {
    expect(feedbackScope("tenant-id","project-id","false","manuals")).toBe("tenant_id=tenant-id&project_id=project-id&limit=100&relevant=false&collection=manuals");
  });
});
