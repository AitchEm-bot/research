# Social Media Accounts & Download Tools

**C2PA Robustness Testing Project - Phase 2.5 Platform Testing**

This document records the social media accounts and third-party download tools used for Phase 2.5 platform round-trip testing.

---

## Associated Social Media Accounts

The following accounts were created for C2PA platform persistence testing:

### Instagram
- **URL**: https://www.instagram.com/independant_researcher/
- **Account Name**: @independant_researcher
- **Content**: 25 images + 10 videos uploaded
- **Download Method**: Third-party downloader (FastDL)

### Twitter (X)
- **URL**: https://x.com/Independant_R
- **Account Name**: @Independant_R
- **Content**: 25 images + 10 videos uploaded
- **Download Method**: Third-party downloader (Snaplytics)

### Facebook
- **URL**: https://www.facebook.com/profile.php?id=61583369476134
- **Profile ID**: 61583369476134
- **Content**: 25 images + 10 videos uploaded
- **Download Method**: Direct download from platform

### YouTube
- **URL**: https://www.youtube.com/channel/UCOwAw40mtHxcMLZG7HyL7fw
- **Channel ID**: UCOwAw40mtHxcMLZG7HyL7fw
- **Content**: 10 videos uploaded (not as Shorts, standard uploads)
- **Download Method**: Direct download from platform

### TikTok
- **URL**: https://www.tiktok.com/@independant_researcher
- **Account Name**: @independant_researcher
- **Content**: 10 videos uploaded
- **Download Method**: Third-party downloader (SnapTik)

### WhatsApp
- **Note**: Not included in this documentation due to lack of message sharing and personal privacy considerations
- **Status**: Downloads not completed

---

## Third-Party Download Tools

Due to platform restrictions, the following third-party tools were used to download media:

### Twitter Media Downloader
- **Service**: Snaplytics
- **URL**: https://snaplytics.io/twitter-img-downloader/
- **Purpose**: Download images and videos from Twitter/X posts
- **Usage**: Paste tweet URL to extract media

### Instagram Media Downloader
- **Service**: FastDL
- **URL**: https://fastdl.app/en2
- **Purpose**: Download images and videos from Instagram posts
- **Usage**: Paste post URL to extract media

### TikTok Downloader
- **Service**: SnapTik
- **URL**: https://snaptik.cx/
- **Purpose**: Download videos from TikTok
- **Usage**: Paste TikTok video URL to extract media

---

## Platform Testing Observations

### Instagram PNG → JPEG Conversion
- **Issue**: Instagram converts all uploaded PNG images to JPEG format
- **Impact**: Lossy compression applied by platform
- **Mitigation**: Documented in platform_results.csv and final analysis
- **Files affected**: All 25 Instagram image uploads
- **Original format**: PNG (1024×1024)
- **Returned format**: JPEG (platform-determined compression quality)

### YouTube Upload Mode
- **Issue**: Videos were uploaded as standard YouTube uploads, not YouTube Shorts
- **Reason**: Platform did not automatically categorize uploads as Shorts
- **Documentation**: Platform folder renamed from `youtube_shorts` to `youtube`, upload mode set to `upload`

### Third-Party Downloader Reliability
- **Quality**: Downloads appear to preserve platform-processed quality
- **Metadata**: C2PA manifests were checked post-download
- **Verification**: All downloaded files were verified using `process_platform_returns.py`

---

## Research Ethics & Privacy

- All accounts were created specifically for this research project
- No personal data or private individuals are depicted in the uploaded media
- All media is AI-generated using:
  - **Images**: Stable Diffusion v1.4 (100 images, seeds 42-141)
  - **Videos**: Google Veo3.1 (60 external videos)
- Accounts are publicly accessible for research verification purposes
- WhatsApp testing was excluded to maintain personal privacy

---

## File Naming Convention

Downloaded files were renamed to follow the standardized format:

```
{original}__{platform}__{mode}__{timestamp}.{ext}
```

**Example**:
```
img_003_seed45_20251113_233209_signed__instagram__post__20251116-122813.jpg
```

**Platform Modes**:
- Instagram: `post`
- Twitter: `upload`
- Facebook: `post`
- YouTube: `upload`
- TikTok: `upload`
- WhatsApp: `compressed` (not tested)

**Timestamp**: 20251116-122813 (YYYYMMDD-HHMMSS format, applied uniformly to all files)

---

## References

- **Project Documentation**: See `CLAUDE.md` for full project context
- **Upload Tracking**: `data/platform_tests/auto_sample_tracking.csv`
- **Rename Log**: `data/platform_tests/rename_log.csv`
- **Platform Results**: `data/results/platform_results.csv` (generated after processing)

---

**Last Updated**: 2025-01-16
**Project**: Is C2PA's Metadata Robust in AI-Generated Content?
**Researcher**: AitchEm
