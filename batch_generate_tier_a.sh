#!/bin/bash
# Batch video generation using Wan 2.2

WAN_DIR=/path/to/Wan
cd $WAN_DIR

# Prompt 0: TIER_A_1
python -m inference.generate --prompt "One realistic human hand palm facing camera with five clearly visible fingers. Fixed white background. Neutral skin tone. No motion. Still pose. Natural skin texture. No jewelry no gloves no text." --negative_prompt "Extra fingers six fingers fused fingers missing fingers malformed hand distorted hand warped skin changing finger count duplicate hand text watermark gloves jewelry camera shake hand rotation" --duration 3 --fps 30 --output videos/no_tremor_still/000.mp4

# Prompt 1: TIER_A_2
python -m inference.generate --prompt "Close-up of one realistic human hand palm facing camera five fingers visible. Neutral white background. Slow smooth horizontal hand drift. No tremor. No shaking. Natural lighting. Clear skin." --negative_prompt "Extra fingers six fingers merged fingers missing fingers hand deformation distorted hand skin melting changing finger count duplicate hand text watermark gloves jewelry sudden hand rotation motion blur" --duration 3 --fps 30 --output videos/no_tremor_smooth_macro/001.mp4

# Prompt 2: TIER_A_3
python -m inference.generate --prompt "One human hand palm down on flat surface fingers relaxed. Five fingers clearly separated. Fixed camera overhead view. No motion. Still pose. Neutral background. Good contrast hand edges clearly defined." --negative_prompt "Extra fingers six fingers fused fingers missing fingers malformed hand distorted hand warped skin changing finger count duplicate hand text watermark gloves jewelry camera shake" --duration 3 --fps 30 --output videos/no_tremor_still/002.mp4

# Prompt 3: TIER_A_4
python -m inference.generate --prompt "Hand remains completely still. Five fingers visible and distinct. No tremor. No shaking. Neutral white background. Fixed camera." --negative_prompt "Extra fingers six fingers merged fingers missing fingers malformed hand distorted hand skin melting changing finger count duplicate hand text watermark gloves jewelry camera movement" --duration 3 --fps 30 --output videos/no_tremor_still/003.mp4

# Prompt 4: TIER_A_5
python -m inference.generate --prompt "Human hand with five fingers clearly visible palm facing camera. Hand resting naturally on neutral surface. Completely still no movement no tremor. Clear skin texture natural lighting." --negative_prompt "Extra fingers six fingers merged fingers missing fingers malformed hand distorted hand skin warping changing finger count duplicate hand text watermark gloves jewelry camera shake motion" --duration 3 --fps 30 --output videos/no_tremor_still/004.mp4

