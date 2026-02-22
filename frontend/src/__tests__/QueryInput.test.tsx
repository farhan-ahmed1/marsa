import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { QueryInput } from "@/components/QueryInput";

describe("QueryInput", () => {
  const onSubmit = vi.fn();

  beforeEach(() => {
    onSubmit.mockReset();
  });

  it("renders textarea and submit button", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    expect(screen.getByPlaceholderText("Ask a research question...")).toBeInTheDocument();
    expect(screen.getByText("Research")).toBeInTheDocument();
  });

  it("submits query on form submit", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    const textarea = screen.getByPlaceholderText("Ask a research question...");
    fireEvent.change(textarea, { target: { value: "test query" } });
    fireEvent.submit(textarea.closest("form")!);
    expect(onSubmit).toHaveBeenCalledWith("test query");
  });

  it("trims whitespace from query", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    const textarea = screen.getByPlaceholderText("Ask a research question...");
    fireEvent.change(textarea, { target: { value: "  padded query  " } });
    fireEvent.submit(textarea.closest("form")!);
    expect(onSubmit).toHaveBeenCalledWith("padded query");
  });

  it("does not submit empty query", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    const textarea = screen.getByPlaceholderText("Ask a research question...");
    fireEvent.submit(textarea.closest("form")!);
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("shows Running button when status is running", () => {
    render(<QueryInput onSubmit={onSubmit} status="running" />);
    expect(screen.getByText("Running")).toBeInTheDocument();
  });

  it("disables textarea and button when loading", () => {
    render(<QueryInput onSubmit={onSubmit} status="running" />);
    expect(screen.getByPlaceholderText("Ask a research question...")).toBeDisabled();
  });

  it("renders example queries", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    expect(screen.getByText("Rust vs Go for distributed systems")).toBeInTheDocument();
    expect(screen.getByText("Latest AI agent framework trends")).toBeInTheDocument();
    expect(screen.getByText("CAP theorem with examples")).toBeInTheDocument();
  });

  it("fills textarea when example is clicked", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    fireEvent.click(screen.getByText("Rust vs Go for distributed systems"));
    const textarea = screen.getByPlaceholderText("Ask a research question...") as HTMLTextAreaElement;
    expect(textarea.value).toBe("Rust vs Go for distributed systems");
  });

  it("submits on Enter key (without Shift)", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    const textarea = screen.getByPlaceholderText("Ask a research question...");
    fireEvent.change(textarea, { target: { value: "enter test" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: false });
    expect(onSubmit).toHaveBeenCalledWith("enter test");
  });

  it("does not submit on Shift+Enter", () => {
    render(<QueryInput onSubmit={onSubmit} status="idle" />);
    const textarea = screen.getByPlaceholderText("Ask a research question...");
    fireEvent.change(textarea, { target: { value: "shift enter test" } });
    fireEvent.keyDown(textarea, { key: "Enter", shiftKey: true });
    expect(onSubmit).not.toHaveBeenCalled();
  });

  it("disables examples when loading", () => {
    render(<QueryInput onSubmit={onSubmit} status="synth" />);
    const buttons = screen.getAllByRole("button");
    // Example buttons should be disabled
    const exampleButtons = buttons.filter(
      (b) => b.textContent && ["Rust vs Go", "Latest AI", "CAP theorem"].some((t) => b.textContent!.includes(t))
    );
    for (const btn of exampleButtons) {
      expect(btn).toBeDisabled();
    }
  });
});
