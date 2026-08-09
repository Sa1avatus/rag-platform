import {describe, expect, it} from "vitest";
import {auditQuery, settingValue} from "./pages";

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
