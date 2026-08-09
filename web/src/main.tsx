import React from "react";
import {createRoot} from "react-dom/client";
import {QueryClient, QueryClientProvider} from "@tanstack/react-query";
import {BrowserRouter} from "react-router-dom";
import {App} from "./App";
import {ErrorBoundary} from "./ErrorBoundary";
import "./styles.css";
createRoot(document.getElementById("root")!).render(<React.StrictMode><ErrorBoundary><QueryClientProvider client={new QueryClient()}><BrowserRouter><App/></BrowserRouter></QueryClientProvider></ErrorBoundary></React.StrictMode>);
