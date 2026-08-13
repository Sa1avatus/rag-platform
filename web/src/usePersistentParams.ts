import {useSearchParams} from "react-router-dom";
import {useCallback, useEffect, useRef} from "react";

const STORAGE_PREFIX = "rag-params:";

type ParamInput = URLSearchParams | string | Record<string, string | string[]>;

/**
 * Drop-in replacement for useSearchParams that persists the last known
 * non-empty values per page in localStorage. On mount, if the URL has no
 * search params, the saved snapshot is restored.
 */
export function usePersistentParams(pageKey: string) {
  const [params, rawSetParams] = useSearchParams();
  const hydrated = useRef(false);

  // On first mount: if URL has no params, restore from localStorage
  useEffect(() => {
    if (hydrated.current) return;
    hydrated.current = true;
    if ([...params.keys()].length > 0) return;
    const saved = localStorage.getItem(STORAGE_PREFIX + pageKey);
    if (!saved) return;
    try {
      const parsed: Record<string, string> = JSON.parse(saved);
      const next = new URLSearchParams(parsed);
      if ([...next.keys()].length > 0) {
        rawSetParams(next, {replace: true});
      }
    } catch { /* corrupt storage – ignore */ }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const setParams = useCallback(
    (next: ParamInput | ((prev: URLSearchParams) => ParamInput), opts?: {replace?: boolean}) => {
      rawSetParams((prev) => {
        const resolved = typeof next === "function" ? next(prev) : next;
        const asUSP = resolved instanceof URLSearchParams
          ? resolved
          : new URLSearchParams(
              typeof resolved === "string"
                ? resolved
                : Object.entries(resolved).flatMap(([k, v]) =>
                    Array.isArray(v) ? v.map(val => [k, val] as [string, string]) : [[k, v] as [string, string]]
                  ),
            );
        // Persist non-empty snapshots
        const entries: Record<string, string> = {};
        asUSP.forEach((v, k) => { entries[k] = v; });
        if (Object.keys(entries).length > 0) {
          localStorage.setItem(STORAGE_PREFIX + pageKey, JSON.stringify(entries));
        }
        return asUSP;
      }, opts);
    },
    [rawSetParams, pageKey],
  );

  return [params, setParams] as const;
}
