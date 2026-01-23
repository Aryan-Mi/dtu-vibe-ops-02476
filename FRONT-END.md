# Vibe-Opsy Frontend

## Overview

The **Vibe-Opsy Frontend** is a retro-futuristic 3D web application that provides an interactive interface for skin lesion analysis. Built with React, Three.js, and modern web technologies, it features a nostalgic 1991 Macintosh Classic computer rendered in 3D where users can upload dermatoscopic images and receive AI-powered diagnostic results.

## Key Features

- **Interactive 3D Macintosh Classic** - Fully rendered vintage computer with authentic CRT screen effects
- **Retro UI Experience** - Mac OS System 7, Matrix terminal, and Windows 95 inspired interfaces
- **Real-time Analysis** - Upload images and receive instant classification results with confidence scores
- **Thermal Receipt Results** - Vintage receipt-style output with drag-to-dismiss interaction
- **Nostalgic Effects** - Dial-up sounds, CRT scanlines, and dot-matrix typography

## Tech Stack

- React 19 + TypeScript
- Three.js + React Three Fiber
- Framer Motion for animations
- Tailwind CSS 4
- Vite 7
- Deployed on Cloudflare Workers

## Live Demo

🌐 **https://vibe-opsy.aryan-mi.workers.dev**

## Repository & Documentation

For complete documentation, installation instructions, and source code:

📂 **https://github.com/Aryan-Mi/vibe-opsy**

## API Integration

The frontend integrates with this backend's inference API:

- Endpoint: `POST /inference`
- Payload: Image file (multipart/form-data)
- Response: Classification results with probabilities
