# 🚀 Render.com Deployment Guide for AI Car Recommendation Backend

## 📋 Prerequisites
- Render.com account (free tier available)
- GitHub repository with your code
- `cars.csv` file in your repository

## 🎯 Quick Deployment Steps

### Step 1: Prepare Your Repository

Ensure your repository has these files:
```
car-rec-backend/
├── app.py                    # Main application
├── requirements.txt          # Dependencies
├── gunicorn.conf.py         # Server configuration
├── render.yaml              # Render configuration
├── cars.csv                 # Car dataset (1.2MB)
└── .gitignore              # Git ignore file
```

### Step 2: Update CORS Origins

**IMPORTANT**: Update the `ALLOWED_ORIGINS` in `render.yaml` with your frontend URL:

```yaml
- key: ALLOWED_ORIGINS
  value: https://your-frontend-domain.com,http://localhost:5173
```

### Step 3: Deploy on Render

#### Method A: Using render.yaml (Recommended)

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Add Render deployment config"
   git push origin main
   ```

2. **Connect to Render**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Blueprint"
   - Connect your GitHub repository
   - Render will automatically detect `render.yaml`

3. **Deploy**
   - Click "Create Blueprint Instance"
   - Render will build and deploy automatically

#### Method B: Manual Setup

1. **Create New Web Service**
   - Go to [render.com](https://render.com)
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

2. **Configure Service**
   ```
   Name: ai-car-recommendation-backend
   Environment: Python
   Build Command: pip install -r requirements.txt
   Start Command: gunicorn --config gunicorn.conf.py app:app
   ```

3. **Set Environment Variables**
   ```
   HOST=0.0.0.0
   PORT=8000
   WORKERS=1
   ALLOWED_ORIGINS=https://your-frontend-domain.com,http://localhost:5173
   EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
   TOP_K_RECOMMENDATIONS=10
   SIMILARITY_THRESHOLD=0.3
   LOG_LEVEL=INFO
   DEBUG=false
   CACHE_EMBEDDINGS=true
   CACHE_FILE=car_embeddings_cache.pkl
   ```

4. **Deploy**
   - Click "Create Web Service"
   - Wait for build to complete (5-10 minutes)

## 🔧 Configuration Details

### Environment Variables Explained

| Variable | Value | Description |
|----------|-------|-------------|
| `HOST` | `0.0.0.0` | Bind to all interfaces |
| `PORT` | `8000` | Render's default port |
| `WORKERS` | `1` | Single worker for free tier |
| `ALLOWED_ORIGINS` | Your frontend URL | CORS configuration |
| `EMBEDDING_MODEL` | `sentence-transformers/all-mpnet-base-v2` | AI model |
| `DEBUG` | `false` | Production mode |

### Resource Requirements

**Free Tier Limits:**
- **RAM**: 512MB (sufficient for model)
- **CPU**: Shared
- **Storage**: 1GB
- **Sleep**: After 15 minutes of inactivity

**Paid Tier Benefits:**
- **RAM**: 1GB+ (faster model loading)
- **CPU**: Dedicated
- **Always On**: No sleep mode

## 📊 Monitoring & Testing

### 1. Check Deployment Status
- Go to your service dashboard on Render
- Check "Logs" tab for build/deployment status
- Monitor "Metrics" for performance

### 2. Test Your API
```bash
# Health check
curl https://your-app-name.onrender.com/

# Test recommendation
curl -X POST "https://your-app-name.onrender.com/recommend" \
  -H "Content-Type: application/json" \
  -d '{"query": "SUV under 10 lakhs"}'
```

### 3. API Documentation
Visit: `https://your-app-name.onrender.com/docs`

## 🔄 Update Frontend Configuration

Update your frontend to use the Render URL:

```javascript
// In your frontend API configuration
const API_BASE_URL = 'https://your-app-name.onrender.com';

// Update fetch calls
const response = await fetch(`${API_BASE_URL}/recommend`, {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({ query: inputMessage })
});
```

## 🚨 Common Issues & Solutions

### 1. Build Failures
**Issue**: Build times out or fails
**Solution**: 
- Check `requirements.txt` has correct versions
- Ensure `cars.csv` is in repository
- Monitor build logs for specific errors

### 2. Model Loading Issues
**Issue**: Service fails to start due to memory
**Solution**:
- Use smaller model: `sentence-transformers/all-MiniLM-L6-v2`
- Upgrade to paid tier for more RAM

### 3. CORS Errors
**Issue**: Frontend can't connect to backend
**Solution**:
- Update `ALLOWED_ORIGINS` with exact frontend URL
- Include both HTTP and HTTPS versions

### 4. Cold Start Delays
**Issue**: First request is slow
**Solution**:
- This is normal for free tier (sleep mode)
- Consider paid tier for always-on service

## 📈 Performance Optimization

### For Free Tier:
```yaml
# Use smaller model
EMBEDDING_MODEL: sentence-transformers/all-MiniLM-L6-v2
WORKERS: 1
```

### For Paid Tier:
```yaml
# Use larger model
EMBEDDING_MODEL: sentence-transformers/all-mpnet-base-v2
WORKERS: 2
```

## 🔐 Security Best Practices

1. **Environment Variables**: Never commit sensitive data
2. **CORS**: Only allow necessary origins
3. **HTTPS**: Render provides automatic SSL
4. **Logs**: Monitor for suspicious activity

## 📞 Support

### Render Support:
- [Render Documentation](https://render.com/docs)
- [Community Forum](https://community.render.com)

### Troubleshooting:
1. Check build logs in Render dashboard
2. Verify environment variables
3. Test API endpoints manually
4. Check CORS configuration

## ✅ Deployment Checklist

- [ ] Repository pushed to GitHub
- [ ] `render.yaml` configured
- [ ] CORS origins updated
- [ ] Environment variables set
- [ ] Build successful
- [ ] Health check passing
- [ ] API endpoints working
- [ ] Frontend updated with new URL
- [ ] Performance tested

## 🎉 Success!

Your AI car recommendation backend is now deployed on Render.com!

**Your API URL**: `https://your-app-name.onrender.com`
**Documentation**: `https://your-app-name.onrender.com/docs`

Remember to update your frontend with the new backend URL! 