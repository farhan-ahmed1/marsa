"""Source Quality Scoring Module.

Implements heuristic-based scoring of information sources to help prioritize
high-quality, authoritative content during research synthesis.

Scoring methodology:
- Domain authority (40%): Based on TLD and known high-quality domains
- Recency (30%): Based on publication date  
- Content depth (30%): Based on word count

Usage:
    from agents.source_scorer import score_source
    
    score = score_source(
        url="https://docs.python.org/3/library/asyncio.html",
        content="... long article content ...",
        published_date="2026-01-15"
    )
    
    print(f"Source score: {score.final_score:.2f}")
    print(f"Domain rating: {score.domain_category}")
"""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

# Add backend directory to path for imports
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from pydantic import BaseModel, Field # noqa: E402
 
from utils.resilience import get_logger # noqa: E402


# Initialize structured logging
logger = get_logger("agents.source_scorer")


# Scoring weights
WEIGHT_DOMAIN = 0.4
WEIGHT_RECENCY = 0.3
WEIGHT_DEPTH = 0.3


# Domain authority categories and their base scores
# Higher scores indicate more authoritative sources

# Government and educational domains are highest authority
DOMAIN_SCORES_TLD = {
    ".gov": 0.9,
    ".edu": 0.9,
    ".gov.uk": 0.9,
    ".ac.uk": 0.9,  # UK academic
    ".edu.au": 0.9,
    ".gov.au": 0.9,
}

# Known high-quality publication domains
DOMAIN_SCORES_PUBLICATIONS = {
    # Academic and research
    "arxiv.org": 0.85,
    "acm.org": 0.85,
    "dl.acm.org": 0.85,
    "ieee.org": 0.85,
    "ieeexplore.ieee.org": 0.85,
    "nature.com": 0.85,
    "science.org": 0.85,
    "sciencedirect.com": 0.85,
    "springer.com": 0.85,
    "link.springer.com": 0.85,
    "pubmed.ncbi.nlm.nih.gov": 0.85,
    "ncbi.nlm.nih.gov": 0.85,
    "scholar.google.com": 0.80,
    "researchgate.net": 0.75,
    
    # Major news outlets
    "nytimes.com": 0.75,
    "washingtonpost.com": 0.75,
    "theguardian.com": 0.75,
    "bbc.com": 0.75,
    "bbc.co.uk": 0.75,
    "reuters.com": 0.80,
    "apnews.com": 0.80,
    
    # Tech news
    "wired.com": 0.70,
    "arstechnica.com": 0.70,
    "techcrunch.com": 0.65,
    "theverge.com": 0.65,
}

# Official documentation and established industry sources
DOMAIN_SCORES_OFFICIAL = {
    # Programming language official docs
    "docs.python.org": 0.85,
    "go.dev": 0.85,
    "golang.org": 0.85,
    "doc.rust-lang.org": 0.85,
    "rust-lang.org": 0.85,
    "docs.microsoft.com": 0.80,
    "learn.microsoft.com": 0.80,
    "developer.apple.com": 0.80,
    "developer.android.com": 0.80,
    "developer.mozilla.org": 0.85,  # MDN is highly authoritative
    "mdn.io": 0.85,
    "nodejs.org": 0.80,
    "typescriptlang.org": 0.80,
    "kotlinlang.org": 0.80,
    "scala-lang.org": 0.80,
    
    # Cloud providers
    "docs.aws.amazon.com": 0.80,
    "aws.amazon.com": 0.75,
    "cloud.google.com": 0.80,
    "azure.microsoft.com": 0.80,
    "docs.oracle.com": 0.80,
    
    # Framework/library docs
    "reactjs.org": 0.80,
    "react.dev": 0.80,
    "vuejs.org": 0.80,
    "angular.io": 0.80,
    "nextjs.org": 0.80,
    "django-project.com": 0.80,
    "flask.palletsprojects.com": 0.80,
    "fastapi.tiangolo.com": 0.80,
    "spring.io": 0.80,
    "kubernetes.io": 0.80,
    "docker.com": 0.75,
    "docs.docker.com": 0.80,
    
    # Databases
    "postgresql.org": 0.80,
    "mysql.com": 0.80,
    "mongodb.com": 0.75,
    "redis.io": 0.80,
    "cassandra.apache.org": 0.80,
    
    # AI/ML
    "openai.com": 0.80,
    "anthropic.com": 0.80,
    "huggingface.co": 0.75,
    "pytorch.org": 0.80,
    "tensorflow.org": 0.80,
    "langchain.com": 0.75,
    "python.langchain.com": 0.75,
}

# Established tech blogs and community sites
DOMAIN_SCORES_BLOGS = {
    # High-quality tech blogs
    "martinfowler.com": 0.75,
    "blog.golang.org": 0.80,
    "rust-lang.github.io": 0.75,
    "engineering.fb.com": 0.70,
    "engineering.atspotify.com": 0.70,
    "netflixtechblog.com": 0.70,
    "instagram-engineering.com": 0.70,
    "engineering.linkedin.com": 0.70,
    "blog.twitter.com": 0.65,
    "uber.com/blog": 0.70,
    "stripe.com/blog": 0.70,
    "discord.com/blog": 0.65,
    
    # Developer communities
    "stackoverflow.com": 0.65,
    "github.com": 0.65,  # READMEs and discussions
    "dev.to": 0.55,
    "medium.com": 0.50,  # Quality varies significantly
    "hashnode.dev": 0.55,
    
    # Tutorial sites
    "realpython.com": 0.70,
    "geeksforgeeks.org": 0.55,
    "tutorialspoint.com": 0.50,
    "w3schools.com": 0.50,
    "freecodecamp.org": 0.60,
    "baeldung.com": 0.65,  # Java/Spring focused
}


class SourceScore(BaseModel):
    """Detailed source quality score breakdown.
    
    Provides transparency into how the final score was calculated,
    allowing agents to understand and explain source quality.
    """
    
    url: str = Field(description="The source URL")
    domain: str = Field(description="Extracted domain from URL")
    final_score: float = Field(
        description="Final weighted score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Component scores
    domain_score: float = Field(
        description="Domain authority score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    recency_score: float = Field(
        description="Recency score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    depth_score: float = Field(
        description="Content depth score (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Metadata
    domain_category: str = Field(
        description="Category of the domain (government, academic, publication, official_docs, blog, unknown)"
    )
    word_count: int = Field(
        description="Word count of the content"
    )
    published_date: Optional[str] = Field(
        default=None,
        description="Published date if provided"
    )
    days_since_published: Optional[int] = Field(
        default=None,
        description="Days since publication if date was provided"
    )

    def to_quality_level(self) -> str:
        """Convert final score to a human-readable quality level."""
        if self.final_score >= 0.8:
            return "high"
        elif self.final_score >= 0.6:
            return "medium"
        elif self.final_score >= 0.4:
            return "low"
        else:
            return "very_low"


def _extract_domain(url: str) -> str:
    """Extract the domain from a URL.
    
    Args:
        url: The URL to extract domain from
        
    Returns:
        The domain (e.g., "docs.python.org")
    """
    if not url:
        return ""
    
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        
        # Remove www. prefix if present
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Remove port if present
        if ":" in domain:
            domain = domain.split(":")[0]
        
        return domain
    except Exception:
        return ""


def _get_domain_score(domain: str, url: str) -> tuple[float, str]:
    """Calculate domain authority score.
    
    Args:
        domain: The extracted domain
        url: The full URL (for path-based matching)
        
    Returns:
        Tuple of (score, category)
    """
    if not domain:
        return 0.4, "unknown"
    
    # Check TLD-based scores (government, education)
    for tld, score in DOMAIN_SCORES_TLD.items():
        if domain.endswith(tld):
            if ".gov" in tld:
                return score, "government"
            else:
                return score, "academic"
    
    # Check full domain matches against known domains
    # Check publications first (highest authority after gov/edu)
    if domain in DOMAIN_SCORES_PUBLICATIONS:
        return DOMAIN_SCORES_PUBLICATIONS[domain], "publication"
    
    # Check for subdomain matches (e.g., "blog.example.com" -> "example.com")
    domain_parts = domain.split(".")
    if len(domain_parts) > 2:
        parent_domain = ".".join(domain_parts[-2:])
        if parent_domain in DOMAIN_SCORES_PUBLICATIONS:
            return DOMAIN_SCORES_PUBLICATIONS[parent_domain], "publication"
    
    # Check official documentation
    if domain in DOMAIN_SCORES_OFFICIAL:
        return DOMAIN_SCORES_OFFICIAL[domain], "official_docs"
    
    # Check parent domain for official docs
    if len(domain_parts) > 2:
        parent_domain = ".".join(domain_parts[-2:])
        if parent_domain in DOMAIN_SCORES_OFFICIAL:
            return DOMAIN_SCORES_OFFICIAL[parent_domain], "official_docs"
    
    # Check established blogs
    if domain in DOMAIN_SCORES_BLOGS:
        return DOMAIN_SCORES_BLOGS[domain], "blog"
    
    # Check parent domain for blogs
    if len(domain_parts) > 2:
        parent_domain = ".".join(domain_parts[-2:])
        if parent_domain in DOMAIN_SCORES_BLOGS:
            return DOMAIN_SCORES_BLOGS[parent_domain], "blog"
    
    # Check URL path for known patterns (e.g., github.com/org/repo)
    if domain == "github.com":
        # GitHub repos from known orgs could get higher scores
        # For now, treat all GitHub content equally
        return 0.65, "blog"
    
    # Default score for unknown domains
    return 0.4, "unknown"


def _get_recency_score(published_date: Optional[str]) -> tuple[float, Optional[int]]:
    """Calculate recency score based on publication date.
    
    Args:
        published_date: Publication date in various formats (ISO, YYYY-MM-DD, etc.)
        
    Returns:
        Tuple of (score, days_since_published)
    """
    if not published_date:
        # No date provided - assume medium recency
        return 0.6, None
    
    # Try to parse the date
    parsed_date = None
    date_formats = [
        "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO with microseconds
        "%Y-%m-%dT%H:%M:%SZ",     # ISO
        "%Y-%m-%dT%H:%M:%S%z",    # ISO with timezone
        "%Y-%m-%dT%H:%M:%S",      # ISO without Z
        "%Y-%m-%d",               # Simple date
        "%B %d, %Y",              # "January 15, 2026"
        "%b %d, %Y",              # "Jan 15, 2026"
        "%d %B %Y",               # "15 January 2026"
        "%d %b %Y",               # "15 Jan 2026"
        "%m/%d/%Y",               # US format
        "%d/%m/%Y",               # EU format
    ]
    
    for fmt in date_formats:
        try:
            parsed_date = datetime.strptime(published_date.strip(), fmt)
            break
        except ValueError:
            continue
    
    if parsed_date is None:
        # Could not parse date - assume medium recency
        logger.debug(
            "could_not_parse_date",
            published_date=published_date,
        )
        return 0.6, None
    
    # Make timezone-aware for comparison
    if parsed_date.tzinfo is None:
        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
    
    now = datetime.now(timezone.utc)
    days_old = (now - parsed_date).days
    
    # Score based on age
    if days_old < 0:
        # Future date (likely error) - treat as very recent
        return 1.0, 0
    elif days_old <= 90:  # 3 months
        return 1.0, days_old
    elif days_old <= 180:  # 6 months
        return 0.8, days_old
    elif days_old <= 365:  # 1 year
        return 0.6, days_old
    else:
        return 0.4, days_old


def _get_depth_score(content: str) -> tuple[float, int]:
    """Calculate content depth score based on word count.
    
    Args:
        content: The content text
        
    Returns:
        Tuple of (score, word_count)
    """
    if not content:
        return 0.4, 0
    
    # Count words (simple whitespace split)
    words = content.split()
    word_count = len(words)
    
    # Score based on word count
    if word_count > 2000:
        return 0.8, word_count
    elif word_count > 500:
        return 0.6, word_count
    else:
        return 0.4, word_count


def score_source(
    url: str,
    content: str,
    published_date: Optional[str] = None,
) -> SourceScore:
    """Calculate quality score for an information source.
    
    Uses a weighted combination of:
    - Domain authority (40%): Based on TLD and known high-quality domains
    - Recency (30%): Based on publication date
    - Content depth (30%): Based on word count
    
    Args:
        url: The source URL
        content: The content from the source
        published_date: Optional publication date (various formats supported)
        
    Returns:
        SourceScore with detailed breakdown
        
    Example:
        >>> score = score_source(
        ...     url="https://docs.python.org/3/library/asyncio.html",
        ...     content="...",
        ...     published_date="2026-01-15"
        ... )
        >>> print(f"Score: {score.final_score:.2f}, Category: {score.domain_category}")
    """
    # Extract domain
    domain = _extract_domain(url)
    
    # Calculate component scores
    domain_score, domain_category = _get_domain_score(domain, url)
    recency_score, days_since = _get_recency_score(published_date)
    depth_score, word_count = _get_depth_score(content)
    
    # Calculate weighted final score
    final_score = (
        WEIGHT_DOMAIN * domain_score +
        WEIGHT_RECENCY * recency_score +
        WEIGHT_DEPTH * depth_score
    )
    
    # Ensure final score is in valid range
    final_score = max(0.0, min(1.0, final_score))
    
    result = SourceScore(
        url=url,
        domain=domain,
        final_score=round(final_score, 4),
        domain_score=domain_score,
        recency_score=recency_score,
        depth_score=depth_score,
        domain_category=domain_category,
        word_count=word_count,
        published_date=published_date,
        days_since_published=days_since,
    )
    
    logger.debug(
        "source_scored",
        url=url[:100],
        domain=domain,
        final_score=result.final_score,
        domain_score=domain_score,
        recency_score=recency_score,
        depth_score=depth_score,
        domain_category=domain_category,
        word_count=word_count,
    )
    
    return result


def score_sources(
    sources: list[dict],
) -> list[SourceScore]:
    """Score multiple sources at once.
    
    Args:
        sources: List of dicts with keys: url, content, published_date (optional)
        
    Returns:
        List of SourceScore objects, sorted by final_score descending
    """
    scores = []
    for source in sources:
        score = score_source(
            url=source.get("url", ""),
            content=source.get("content", ""),
            published_date=source.get("published_date"),
        )
        scores.append(score)
    
    # Sort by final score, highest first
    scores.sort(key=lambda s: s.final_score, reverse=True)
    
    return scores


def filter_sources_by_quality(
    sources: list[dict],
    min_score: float = 0.5,
) -> list[SourceScore]:
    """Score sources and filter by minimum quality threshold.
    
    Args:
        sources: List of dicts with keys: url, content, published_date (optional)
        min_score: Minimum final_score to include (default: 0.5)
        
    Returns:
        List of SourceScore objects above threshold, sorted by final_score descending
    """
    scores = score_sources(sources)
    return [s for s in scores if s.final_score >= min_score]


# Convenience aliases for common categorization checks
def is_high_quality(score: SourceScore) -> bool:
    """Check if a source is high quality (score >= 0.7)."""
    return score.final_score >= 0.7


def is_authoritative(score: SourceScore) -> bool:
    """Check if a source is from authoritative domain (gov, edu, publication, official_docs)."""
    return score.domain_category in ("government", "academic", "publication", "official_docs")


def is_recent(score: SourceScore) -> bool:
    """Check if a source is recent (within 6 months)."""
    return score.recency_score >= 0.8
