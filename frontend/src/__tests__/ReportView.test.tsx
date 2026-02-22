import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { ReportView } from "@/components/ReportView";
import type { Report, Citation } from "@/lib/types";

// Mock react-markdown since it's an ESM module that doesn't work well in jsdom
vi.mock("react-markdown", () => ({
  default: ({ children }: { children: string }) => <div data-testid="markdown">{children}</div>,
}));

// Mock child components to keep tests focused
vi.mock("@/components/SourceCard", () => ({
  SourceList: ({ citations }: { citations: Citation[] }) => (
    <div data-testid="source-list">{citations.length} sources</div>
  ),
}));

vi.mock("@/components/Metrics", () => ({
  Metrics: ({ metrics }: { metrics: unknown[] }) => (
    <div data-testid="metrics">{metrics.length} items</div>
  ),
}));

const sampleReport: Report = {
  title: "Test Report Title",
  summary: "This is a summary with a citation [1].",
  sections: [
    { heading: "Introduction", content: "Intro content [1] and [2]." },
    { heading: "Analysis", content: "Analysis content." },
  ],
  confidence_summary: "High confidence in findings.",
  citations: [
    {
      number: 1,
      title: "Source A",
      url: "https://example.com/a",
      source_quality_score: 0.9,
      accessed_date: "2025-01-01",
    },
    {
      number: 2,
      title: "Source B",
      url: "https://example.com/b",
      source_quality_score: 0.7,
      accessed_date: "2025-01-02",
    },
  ],
};

describe("ReportView", () => {
  it("shows placeholder when no report", () => {
    render(<ReportView report={null} />);
    expect(screen.getByText("Report will appear here once research is complete.")).toBeInTheDocument();
  });

  it("shows skeleton when loading", () => {
    const { container } = render(<ReportView report={null} isLoading />);
    expect(container.querySelector(".animate-pulse")).toBeInTheDocument();
  });

  it("renders report title", () => {
    render(<ReportView report={sampleReport} />);
    expect(screen.getByText("Test Report Title")).toBeInTheDocument();
  });

  it("renders sections with headings", () => {
    render(<ReportView report={sampleReport} />);
    expect(screen.getByText("Introduction")).toBeInTheDocument();
    expect(screen.getByText("Analysis")).toBeInTheDocument();
  });

  it("renders confidence summary", () => {
    render(<ReportView report={sampleReport} />);
    expect(screen.getByText("High confidence in findings.")).toBeInTheDocument();
  });

  it("renders citation links as buttons", () => {
    render(<ReportView report={sampleReport} />);
    // Citation [1] appears in summary and section, [2] in section
    const citationButtons = screen.getAllByRole("button").filter(
      (b) => b.textContent === "1" || b.textContent === "2"
    );
    expect(citationButtons.length).toBeGreaterThanOrEqual(2);
  });

  it("renders source list with citations", () => {
    render(<ReportView report={sampleReport} />);
    expect(screen.getByTestId("source-list")).toHaveTextContent("2 sources");
  });

  it("shows copy button", () => {
    render(<ReportView report={sampleReport} />);
    expect(screen.getByText("Copy")).toBeInTheDocument();
  });

  it("renders rawReport via markdown when no structured sections", () => {
    const report: Report = {
      title: "",
      summary: "",
      sections: [],
      confidence_summary: "",
      citations: [],
    };
    render(<ReportView report={report} rawReport="# Raw Markdown Report" />);
    expect(screen.getByTestId("markdown")).toHaveTextContent("# Raw Markdown Report");
  });

  it("renders metrics when provided", () => {
    const metrics = [
      { key: "latency", value: "5.2s", sub: "total" },
      { key: "llm_calls", value: "4", sub: "2.1k tok" },
    ];
    render(<ReportView report={sampleReport} metrics={metrics} />);
    expect(screen.getByTestId("metrics")).toHaveTextContent("2 items");
  });

  it("shows Research Report label", () => {
    render(<ReportView report={sampleReport} />);
    expect(screen.getByText("Research Report")).toBeInTheDocument();
  });
});
