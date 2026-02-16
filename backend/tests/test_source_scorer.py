"""Tests for the source quality scoring module.

Tests domain authority scoring, recency scoring, content depth scoring,
and the combined source quality score.
"""

import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent.parent
sys.path.insert(0, str(backend_dir))

from agents.source_scorer import (  # noqa: E402
    score_source,
    score_sources,
    filter_sources_by_quality,
    SourceScore,
    is_high_quality,
    is_authoritative,
    is_recent,
    _extract_domain,
    _get_domain_score,
    _get_recency_score,
    _get_depth_score,
)


class TestExtractDomain:
    """Tests for domain extraction from URLs."""
    
    def test_simple_url(self):
        """Test domain extraction from simple URL."""
        assert _extract_domain("https://example.com/page") == "example.com"
    
    def test_url_with_www(self):
        """Test www prefix is removed."""
        assert _extract_domain("https://www.example.com/page") == "example.com"
    
    def test_url_with_subdomain(self):
        """Test subdomains are preserved."""
        assert _extract_domain("https://docs.python.org/3/") == "docs.python.org"
    
    def test_url_with_port(self):
        """Test port is removed."""
        assert _extract_domain("http://localhost:8000/api") == "localhost"
    
    def test_empty_url(self):
        """Test empty URL returns empty string."""
        assert _extract_domain("") == ""
    
    def test_invalid_url(self):
        """Test invalid URL returns empty string."""
        assert _extract_domain("not-a-url") == ""


class TestDomainScoring:
    """Tests for domain authority scoring."""
    
    def test_gov_domain(self):
        """Test government domains get high score."""
        score, category = _get_domain_score("python.gov", "https://python.gov/docs")
        assert score == 0.9
        assert category == "government"
    
    def test_edu_domain(self):
        """Test educational domains get high score."""
        score, category = _get_domain_score("mit.edu", "https://mit.edu/research")
        assert score == 0.9
        assert category == "academic"
    
    def test_arxiv_domain(self):
        """Test arxiv gets publication score."""
        score, category = _get_domain_score("arxiv.org", "https://arxiv.org/abs/1234")
        assert score == 0.85
        assert category == "publication"
    
    def test_python_docs(self):
        """Test Python official docs gets high score."""
        score, category = _get_domain_score("docs.python.org", "https://docs.python.org/3/")
        assert score == 0.85
        assert category == "official_docs"
    
    def test_medium_blog(self):
        """Test medium gets blog score."""
        score, category = _get_domain_score("medium.com", "https://medium.com/article")
        assert score == 0.50
        assert category == "blog"
    
    def test_stackoverflow(self):
        """Test StackOverflow gets blog score."""
        score, category = _get_domain_score("stackoverflow.com", "https://stackoverflow.com/q/123")
        assert score == 0.65
        assert category == "blog"
    
    def test_unknown_domain(self):
        """Test unknown domains get default score."""
        score, category = _get_domain_score("randomsite123.com", "https://randomsite123.com")
        assert score == 0.4
        assert category == "unknown"
    
    def test_empty_domain(self):
        """Test empty domain returns default score."""
        score, category = _get_domain_score("", "")
        assert score == 0.4
        assert category == "unknown"


class TestRecencyScoring:
    """Tests for recency scoring based on publication date."""
    
    def test_recent_date_3_months(self):
        """Test date within 3 months gets full score."""
        recent_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        score, days = _get_recency_score(recent_date)
        assert score == 1.0
        assert days is not None
        assert days <= 90
    
    def test_date_6_months_old(self):
        """Test date 6 months old gets 0.8 score."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=150)).strftime("%Y-%m-%d")
        score, days = _get_recency_score(old_date)
        assert score == 0.8
        assert days is not None
        assert 90 < days <= 180
    
    def test_date_1_year_old(self):
        """Test date 1 year old gets 0.6 score."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=300)).strftime("%Y-%m-%d")
        score, days = _get_recency_score(old_date)
        assert score == 0.6
        assert days is not None
        assert 180 < days <= 365
    
    def test_date_more_than_1_year(self):
        """Test date older than 1 year gets 0.4 score."""
        old_date = (datetime.now(timezone.utc) - timedelta(days=500)).strftime("%Y-%m-%d")
        score, days = _get_recency_score(old_date)
        assert score == 0.4
        assert days is not None
        assert days > 365
    
    def test_no_date_provided(self):
        """Test missing date returns medium score."""
        score, days = _get_recency_score(None)
        assert score == 0.6
        assert days is None
    
    def test_iso_format_date(self):
        """Test ISO format date parsing."""
        recent_date = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
        score, days = _get_recency_score(recent_date)
        assert score == 1.0
        assert days is not None
    
    def test_invalid_date_format(self):
        """Test invalid date format returns medium score."""
        score, days = _get_recency_score("not-a-date")
        assert score == 0.6
        assert days is None


class TestDepthScoring:
    """Tests for content depth scoring based on word count."""
    
    def test_short_content(self):
        """Test short content gets low score."""
        content = "This is a short article with few words."
        score, word_count = _get_depth_score(content)
        assert score == 0.4
        assert word_count < 500
    
    def test_medium_content(self):
        """Test medium content gets medium score."""
        content = " ".join(["word"] * 600)
        score, word_count = _get_depth_score(content)
        assert score == 0.6
        assert 500 <= word_count <= 2000
    
    def test_long_content(self):
        """Test long content gets high score."""
        content = " ".join(["word"] * 2500)
        score, word_count = _get_depth_score(content)
        assert score == 0.8
        assert word_count > 2000
    
    def test_empty_content(self):
        """Test empty content returns low score."""
        score, word_count = _get_depth_score("")
        assert score == 0.4
        assert word_count == 0
    
    def test_none_content(self):
        """Test None content returns low score."""
        score, word_count = _get_depth_score(None)
        assert score == 0.4
        assert word_count == 0


class TestScoreSource:
    """Tests for the main score_source function."""
    
    def test_high_quality_gov_source(self):
        """Test scoring a high-quality government source."""
        content = " ".join(["word"] * 2500)  # Long content
        recent_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        
        score = score_source(
            url="https://python.gov/docs/tutorial",
            content=content,
            published_date=recent_date
        )
        
        assert isinstance(score, SourceScore)
        assert score.domain_score == 0.9
        assert score.recency_score == 1.0
        assert score.depth_score == 0.8
        # Weighted: 0.4 * 0.9 + 0.3 * 1.0 + 0.3 * 0.8 = 0.36 + 0.3 + 0.24 = 0.9
        assert 0.85 <= score.final_score <= 0.95
        assert score.domain_category == "government"
    
    def test_low_quality_unknown_source(self):
        """Test scoring a low-quality unknown source."""
        content = "Short content."
        
        score = score_source(
            url="https://randomsite789.xyz/article",
            content=content,
            published_date=None
        )
        
        assert score.domain_score == 0.4
        assert score.recency_score == 0.6  # Default for missing date
        assert score.depth_score == 0.4
        # Weighted: 0.4 * 0.4 + 0.3 * 0.6 + 0.3 * 0.4 = 0.16 + 0.18 + 0.12 = 0.46
        assert 0.4 <= score.final_score <= 0.5
        assert score.domain_category == "unknown"
    
    def test_medium_quality_blog_source(self):
        """Test scoring a medium-quality blog source."""
        content = " ".join(["word"] * 800)  # Medium content
        old_date = (datetime.now(timezone.utc) - timedelta(days=200)).strftime("%Y-%m-%d")
        
        score = score_source(
            url="https://realpython.com/article",
            content=content,
            published_date=old_date
        )
        
        assert 0.55 <= score.final_score <= 0.7
        assert score.domain_category == "blog"
    
    def test_quality_level_high(self):
        """Test quality level classification for high score."""
        content = " ".join(["word"] * 2500)
        recent_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        
        score = score_source(
            url="https://docs.python.org/3/tutorial",
            content=content,
            published_date=recent_date
        )
        
        assert score.to_quality_level() == "high"
    
    def test_quality_level_low(self):
        """Test quality level classification for low score."""
        score = score_source(
            url="https://unknown-site.xyz/random",
            content="Short.",
            published_date=None
        )
        
        assert score.to_quality_level() in ("low", "very_low")


class TestScoreSources:
    """Tests for batch source scoring."""
    
    def test_score_multiple_sources(self):
        """Test scoring multiple sources at once."""
        sources = [
            {
                "url": "https://docs.python.org/3/",
                "content": " ".join(["word"] * 1000),
                "published_date": None
            },
            {
                "url": "https://random.com/page",
                "content": "Short content",
                "published_date": None
            },
            {
                "url": "https://arxiv.org/abs/123",
                "content": " ".join(["word"] * 2500),
                "published_date": None
            }
        ]
        
        scores = score_sources(sources)
        
        assert len(scores) == 3
        # Should be sorted by final_score descending
        assert scores[0].final_score >= scores[1].final_score >= scores[2].final_score
    
    def test_filter_by_quality(self):
        """Test filtering sources by minimum quality threshold."""
        sources = [
            {
                "url": "https://docs.python.org/3/",
                "content": " ".join(["word"] * 1000),
                "published_date": None
            },
            {
                "url": "https://random.com/page",
                "content": "Short",
                "published_date": None
            }
        ]
        
        filtered = filter_sources_by_quality(sources, min_score=0.5)
        
        # Only high-quality sources should remain
        assert all(s.final_score >= 0.5 for s in filtered)


class TestHelperFunctions:
    """Tests for helper classification functions."""
    
    def test_is_high_quality(self):
        """Test is_high_quality helper."""
        high_score = SourceScore(
            url="https://test.gov", domain="test.gov", final_score=0.8,
            domain_score=0.9, recency_score=0.8, depth_score=0.7,
            domain_category="government", word_count=1000, published_date=None
        )
        low_score = SourceScore(
            url="https://test.com", domain="test.com", final_score=0.5,
            domain_score=0.4, recency_score=0.6, depth_score=0.5,
            domain_category="unknown", word_count=200, published_date=None
        )
        
        assert is_high_quality(high_score) is True
        assert is_high_quality(low_score) is False
    
    def test_is_authoritative(self):
        """Test is_authoritative helper."""
        gov_score = SourceScore(
            url="https://test.gov", domain="test.gov", final_score=0.8,
            domain_score=0.9, recency_score=0.8, depth_score=0.7,
            domain_category="government", word_count=1000, published_date=None
        )
        blog_score = SourceScore(
            url="https://medium.com", domain="medium.com", final_score=0.5,
            domain_score=0.5, recency_score=0.6, depth_score=0.5,
            domain_category="blog", word_count=500, published_date=None
        )
        
        assert is_authoritative(gov_score) is True
        assert is_authoritative(blog_score) is False
    
    def test_is_recent(self):
        """Test is_recent helper."""
        recent_score = SourceScore(
            url="https://test.com", domain="test.com", final_score=0.6,
            domain_score=0.5, recency_score=1.0, depth_score=0.5,
            domain_category="blog", word_count=500, published_date="2026-02-01",
            days_since_published=15
        )
        old_score = SourceScore(
            url="https://test.com", domain="test.com", final_score=0.5,
            domain_score=0.5, recency_score=0.4, depth_score=0.5,
            domain_category="blog", word_count=500, published_date="2024-01-01",
            days_since_published=400
        )
        
        assert is_recent(recent_score) is True
        assert is_recent(old_score) is False


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_url_and_content(self):
        """Test handling of empty inputs."""
        score = score_source(url="", content="", published_date=None)
        
        # Weighted: 0.4*0.4 + 0.3*0.6 + 0.3*0.4 = 0.46 (recency defaults to 0.6 when no date)
        assert score.final_score == 0.46
        assert score.domain == ""
        assert score.word_count == 0
        assert score.domain_category == "unknown"
    
    def test_special_characters_in_url(self):
        """Test URL with special characters."""
        score = score_source(
            url="https://example.com/path?query=value&foo=bar#section",
            content="Some content here.",
            published_date=None
        )
        
        assert score.domain == "example.com"
    
    def test_unicode_content(self):
        """Test content with unicode characters."""
        content = "Python est un langage de programmation. " * 100
        score = score_source(
            url="https://docs.python.org/fr/",
            content=content,
            published_date=None
        )
        
        assert score.word_count > 0
        assert 0 < score.final_score <= 1.0
