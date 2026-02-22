#!/usr/bin/env bash
# --------------------------------------------------------------------------
# record_demo.sh -- Helper script for recording a MARSA demo GIF / video
#
# Prerequisites:
#   - OBS Studio (for screen recording):  https://obsproject.com
#   - gifski (for high-quality GIF conversion): brew install gifski
#   - ffmpeg (for extracting frames): brew install ffmpeg
#
# Usage:
#   1. Start the stack:      make dev   (or docker-compose up)
#   2. Run this script:      bash scripts/record_demo.sh
#   3. Follow the prompts.
#
# The script can also convert an existing recording to a GIF.
# --------------------------------------------------------------------------

set -euo pipefail

VIDEO_DIR="./docs/assets"
GIF_OUTPUT="$VIDEO_DIR/demo.gif"
FPS=12
WIDTH=960

mkdir -p "$VIDEO_DIR"

# ------------------------------------------------------------------
# Helper: convert an MP4/MOV to an optimised GIF via gifski
# ------------------------------------------------------------------
convert_to_gif() {
    local input="$1"
    local output="${2:-$GIF_OUTPUT}"

    echo "Extracting frames at ${FPS} fps..."
    local tmpdir
    tmpdir=$(mktemp -d)
    ffmpeg -hide_banner -loglevel warning \
        -i "$input" \
        -vf "fps=${FPS},scale=${WIDTH}:-1:flags=lanczos" \
        "$tmpdir/frame%04d.png"

    echo "Encoding GIF with gifski..."
    gifski --fps "$FPS" --width "$WIDTH" --quality 80 \
        -o "$output" "$tmpdir"/frame*.png

    rm -rf "$tmpdir"
    echo "GIF saved to $output ($(du -h "$output" | cut -f1))"
}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
echo "=========================================="
echo " MARSA Demo Recording Helper"
echo "=========================================="
echo ""
echo "Recommended recording flow (60-90 seconds):"
echo ""
echo "  1. Open http://localhost:3000 in your browser"
echo "  2. Type a complex query, e.g.:"
echo "     'Compare Rust vs Go for building distributed systems'"
echo "  3. Show the agent trace streaming in real-time"
echo "  4. Interact with the HITL checkpoint (approve or dig deeper)"
echo "  5. Show the final report with citations and source cards"
echo "  6. Open the observability timeline tab"
echo ""
echo "Options:"
echo "  [1] Open OBS Studio for recording"
echo "  [2] Convert an existing video file to GIF"
echo "  [3] Start backend + frontend for recording"
echo "  [q] Quit"
echo ""

read -rp "Choice: " choice

case "$choice" in
    1)
        echo "Opening OBS Studio..."
        if command -v obs &>/dev/null; then
            obs &
        elif [[ -d "/Applications/OBS.app" ]]; then
            open /Applications/OBS.app
        else
            echo "OBS Studio not found. Install from https://obsproject.com"
            exit 1
        fi
        echo ""
        echo "Tips:"
        echo "  - Set canvas to 1920x1080 or 1280x720"
        echo "  - Use Window Capture for the browser"
        echo "  - Save recording to $VIDEO_DIR/"
        echo "  - After recording, re-run this script with option [2] to convert to GIF"
        ;;
    2)
        read -rp "Path to video file (mp4/mov): " video_path
        if [[ ! -f "$video_path" ]]; then
            echo "File not found: $video_path"
            exit 1
        fi
        if ! command -v ffmpeg &>/dev/null; then
            echo "ffmpeg is required. Install with: brew install ffmpeg"
            exit 1
        fi
        if ! command -v gifski &>/dev/null; then
            echo "gifski is required. Install with: brew install gifski"
            exit 1
        fi
        convert_to_gif "$video_path"
        ;;
    3)
        echo "Starting backend and frontend..."
        make dev
        ;;
    q|Q)
        echo "Bye!"
        exit 0
        ;;
    *)
        echo "Unknown option: $choice"
        exit 1
        ;;
esac
