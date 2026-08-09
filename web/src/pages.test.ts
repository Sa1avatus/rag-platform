import {describe, expect, it} from "vitest";
import {settingValue} from "./pages";

describe("settingValue", () => {
  it("parses boolean controls", () => {
    expect(settingValue("false", true)).toBe(false);
    expect(settingValue("true", false)).toBe(true);
  });

  it("parses numeric controls", () => {
    expect(settingValue("42", 30)).toBe(42);
  });
});
