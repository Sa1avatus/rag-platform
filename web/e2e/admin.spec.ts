import {expect, test, type Page} from "@playwright/test";

const LOCAL_ADMIN_TOKEN = "local-rag-admin-token";

async function signIn(page: Page) {
  await page.goto("/");
  await page.getByRole("textbox", {name: "Admin token"}).fill(LOCAL_ADMIN_TOKEN);
  await page.getByRole("button", {name: "Sign in"}).click();
  await expect(page.getByRole("heading", {name: "Dashboard", level: 1})).toBeVisible();
}

test("admin authentication survives navigation", async ({page}) => {
  await signIn(page);
  await page.getByRole("button", {name: "Use light theme"}).click();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.getByRole("link", {name: "Settings"}).click();
  await expect(page.getByRole("heading", {name: "Settings", level: 1})).toBeVisible();
  await page.reload();
  await expect(page.getByRole("heading", {name: "Settings", level: 1})).toBeVisible();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});

test("every administrative section has a real route", async ({page}) => {
  await signIn(page);
  const routes = [
    ["dashboard", "Dashboard"],
    ["projects", "Projects"],
    ["collections", "Collections"],
    ["documents", "Documents"],
    ["indexing", "Indexing"],
    ["search-playground", "Search Playground"],
    ["retrieval-traces", "Retrieval Traces"],
    ["evaluation", "Evaluation"],
    ["feedback", "Feedback"],
    ["models", "Models"],
    ["reranker", "Reranker"],
    ["system-health", "System Health"],
    ["settings", "Settings"],
    ["audit-log", "Audit Log"],
  ] as const;
  for (const [route, heading] of routes) {
    await page.goto(`/${route}`);
    await expect(page.getByRole("heading", {name: heading, level: 1})).toBeVisible();
    await expect(page.getByRole("textbox", {name: "Admin token"})).toHaveCount(0);
  }
});

test("reranker outage is presented as a safe degraded state", async ({page}) => {
  await page.route("**/api/v1/admin/reranker/status", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {status: "unavailable", error: "ConnectError"},
    });
  });
  await page.route("**/api/v1/admin/reranker/test", async (route) => {
    await route.fulfill({
      contentType: "application/json",
      json: {status: "unavailable", error: "ConnectError"},
    });
  });
  await signIn(page);
  await page.goto("/reranker");
  await expect(page.getByRole("heading", {name: "Reranker", level: 1})).toBeVisible();
  await expect(
    page.getByText(/Retrieval remains available through the configured fallback/)
  ).toBeVisible({timeout: 10_000});
  await page.getByRole("button", {name: "Test connection"}).click();
  await expect(page.getByText(/Safe error:/)).toBeVisible({timeout: 10_000});
});
