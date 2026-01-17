# PhishNetra Deployment Guide

This guide covers deploying PhishNetra to production:
- **Frontend**: Vercel
- **Backend**: Render

## Prerequisites

1. GitHub account with this repository
2. Vercel account (free tier works)
3. Render account (free tier works)

## Step 1: Push to GitHub

```bash
# Initialize git repository (if not already initialized)
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: PhishNetra project"

# Add remote (replace with your GitHub repo URL)
git remote add origin https://github.com/anant720/PhishNetra.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## Step 2: Deploy Backend to Render

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Create New Web Service**:
   - Connect your GitHub repository
   - Select the `PhishNetra` repository
   
3. **Configure the Service**:
   - **Name**: `phishnetra-backend`
   - **Environment**: `Python 3`
   - **Build Command**:
     ```bash
     cd backend && pip install --upgrade pip && pip install -r requirements.txt && python -m spacy download en_core_web_sm
     ```
   - **Start Command**:
     ```bash
     cd backend && uvicorn main:app --host 0.0.0.0 --port $PORT
     ```
   - **Root Directory**: `backend`

4. **Set Environment Variables** (in Render dashboard):
   - `PYTHON_VERSION`: `3.11.0`
   - `ENVIRONMENT`: `production`
   - `API_HOST`: `0.0.0.0`
   - `PORT`: (automatically set by Render, sync with `API_PORT`)
   - `CORS_ORIGINS`: `https://your-vercel-app.vercel.app` (update after deploying frontend)
   - `LOG_LEVEL`: `INFO`
   - `RATE_LIMIT_REQUESTS`: `100`
   - `RATE_LIMIT_WINDOW`: `60`

5. **Deploy**: Click "Create Web Service"

6. **Note your Backend URL**: Render will provide a URL like `https://phishnetra-backend.onrender.com`

## Step 3: Deploy Frontend to Vercel

### Option A: Using Vercel Dashboard (Recommended)

1. **Go to Vercel Dashboard**: https://vercel.com/dashboard
2. **Import Project**:
   - Click "Add New" → "Project"
   - Import your GitHub repository
   - Select the `PhishNetra` repository

3. **Configure Project Settings**:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build` (or leave default)
   - **Output Directory**: `.next` (or leave default)
   - **Install Command**: `npm install`

4. **Set Environment Variables**:
   - `NEXT_PUBLIC_API_BASE_URL`: `https://your-render-backend-url.onrender.com`
     - Replace with your actual Render backend URL from Step 2

5. **Deploy**: Click "Deploy"

### Option B: Using Vercel CLI

```bash
# Install Vercel CLI
npm i -g vercel

# Navigate to frontend directory
cd frontend

# Deploy
vercel

# Set environment variable
vercel env add NEXT_PUBLIC_API_BASE_URL production
# Enter: https://your-render-backend-url.onrender.com
```

## Step 4: Update CORS Settings

After deploying the frontend, update the backend CORS settings:

1. Go to Render dashboard → Your backend service → Environment
2. Update `CORS_ORIGINS` to include your Vercel URL:
   - Example: `https://phishnetra.vercel.app,https://phishnetra-git-main-yourname.vercel.app`

3. Redeploy the backend service

## Step 5: Verify Deployment

1. **Check Backend Health**:
   - Visit: `https://your-backend.onrender.com/api/v1/health`
   - Should return: `{"status": "healthy"}`

2. **Check Frontend**:
   - Visit your Vercel URL
   - Try analyzing a message
   - Check browser console for any errors

## Environment Variables Reference

### Backend (Render)

| Variable | Value | Description |
|----------|-------|-------------|
| `PYTHON_VERSION` | `3.11.0` | Python version |
| `ENVIRONMENT` | `production` | Environment mode |
| `CORS_ORIGINS` | `https://your-app.vercel.app` | Allowed origins |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RATE_LIMIT_REQUESTS` | `100` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

### Frontend (Vercel)

| Variable | Value | Description |
|----------|-------|-------------|
| `NEXT_PUBLIC_API_BASE_URL` | `https://your-backend.onrender.com` | Backend API URL |

## Troubleshooting

### Backend Issues

1. **Build Fails**: Check build logs in Render dashboard
   - Common issues: Missing dependencies, Python version mismatch
   - Solution: Ensure `requirements.txt` is complete

2. **Service Not Starting**: 
   - Check start command includes `--port $PORT`
   - Verify Python path and uvicorn installation

3. **CORS Errors**:
   - Ensure `CORS_ORIGINS` includes your Vercel URL
   - Check for trailing slashes in URLs

### Frontend Issues

1. **Build Fails**:
   - Check `package.json` dependencies
   - Verify Node.js version (requires 18+)

2. **API Calls Fail**:
   - Verify `NEXT_PUBLIC_API_BASE_URL` is set correctly
   - Check browser console for CORS errors
   - Ensure backend is deployed and running

3. **Environment Variables Not Working**:
   - Vercel requires `NEXT_PUBLIC_` prefix for client-side env vars
   - Redeploy after changing environment variables

## Custom Domains

### Vercel Custom Domain

1. Go to Vercel dashboard → Your project → Settings → Domains
2. Add your custom domain
3. Follow DNS configuration instructions

### Render Custom Domain

1. Go to Render dashboard → Your service → Settings → Custom Domains
2. Add your custom domain
3. Update DNS records as instructed

## Monitoring

- **Vercel**: Check deployment logs and analytics in dashboard
- **Render**: View logs in service dashboard
- **Application**: Health endpoint at `/api/v1/health`

## Cost Estimation

- **Vercel (Hobby Plan)**: Free (suitable for most projects)
- **Render (Free Tier)**: Free (with limitations)
  - Services spin down after 15 minutes of inactivity
  - First request after spin-down may take longer

For production workloads, consider Render paid plans for always-on services.

## Support

For issues or questions:
1. Check application logs in respective dashboards
2. Review GitHub issues
3. Check documentation in `/docs` folder
