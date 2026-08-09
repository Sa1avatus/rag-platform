import {Component, type ErrorInfo, type ReactNode} from "react";

type Props = {children: ReactNode};
type State = {failed: boolean};

export class ErrorBoundary extends Component<Props, State> {
  state: State = {failed: false};

  static getDerivedStateFromError(): State {
    return {failed: true};
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error("Admin UI rendering failed", {
      componentStack: info.componentStack,
      error: error.name,
    });
  }

  render() {
    if (this.state.failed) {
      return <main className="fatal-error" role="alert">
        <h1>Unable to render this page</h1>
        <p>The administrative UI encountered an unexpected error. No operation was submitted.</p>
        <button onClick={() => window.location.reload()}>Reload application</button>
      </main>;
    }
    return this.props.children;
  }
}
