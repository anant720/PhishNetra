# Quick Deployment Guide

## ✅ Repository Status
Your project has been pushed to GitHub: https://github.com/anant720/PhishNetra.git

## 🚀 Next Steps

### 1. Deploy Backend to Render (5 minutes)

1. Go to https://dashboard.render.com
2. Click "New +" → "Web Service"
3. Connect GitHub and select `PhishNetra` repository
4. Configure:
   - **Name**: `phishnetra-backend`
   - **Root Directory**: `backend`
   - **Environment**: Python 3
   - **Build Command**: (auto-filled from render.yaml)
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. **Set Environment Variables**:
   - `PYTHON_VERSION`: `3.11.0`
   - `CORS_ORIGINS`: `https://phishnetra.vercel.app` (update after frontend deployment)
6. Click "Create Web Service"
7. **Copy the URL** (e.g., `https://phishnetra-backend.onrender.com`)

### 2. Deploy Frontend to Vercel (3 minutes)

1. Go to https://vercel.com/dashboard
2. Click "Add New" → "Project"
3. Import `PhishNetra` repository from GitHub
4. Configure:
   - **Framework Preset**: Next.js
   - **Root Directory**: `frontend`
   - **Build Command**: `npm run build`
5. **Set Environment Variable**:
   - `NEXT_PUBLIC_API_BASE_URL`: `https://your-render-backend-url.onrender.com`
     - Use the URL from Step 1
6. Click "Deploy"
7. **Copy the URL** (e.g., `https://phishnetra.vercel.app`)

### 3. Update Backend CORS (1 minute)

1. Go back to Render dashboard
2. Your backend service → Environment
3. Update `CORS_ORIGINS` to include your Vercel URL:
   - `https://phishnetra.vercel.app,https://phishnetra-git-main-username.vercel.app`
4. Save changes (auto-deploys)

### 4. Test Deployment

- Backend Health: `https://your-backend.onrender.com/api/v1/health`
- Frontend: Visit your Vercel URL and try analyzing a message

## 📝 Important Notes

- **Render Free Tier**: Services spin down after 15 minutes of inactivity. First request may be slow.
- **Vercel Free Tier**: Excellent for production, includes analytics.
- **Environment Variables**: 
  - Backend: Set in Render dashboard
  - Frontend: Set in Vercel dashboard (must have `NEXT_PUBLIC_` prefix)

## 🔧 Troubleshooting

- **CORS Errors**: Make sure `CORS_ORIGINS` includes your Vercel URL
- **API Not Working**: Check `NEXT_PUBLIC_API_BASE_URL` is set correctly
- **Build Fails**: Check logs in respective dashboards

## 📚 Full Documentation

See `DEPLOYMENT.md` for detailed deployment instructions.
