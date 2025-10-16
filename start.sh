#!/bin/bash

# SkinCare Pro Production Startup Script

set -e  # Exit on error

echo "🚀 Starting SkinCare Pro..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Setup environment
if [ ! -f ".env" ]; then
    echo "⚙️  Creating environment file..."
    cp .env.example .env
    echo "⚠️  Please update .env with your configuration!"
fi

# Initialize database
echo "🗄️  Initializing database..."
cd backend
python -c "from database import init_db; init_db()"

# Build frontend
echo "🎨 Building frontend..."
cd ../frontend
if [ ! -d "node_modules" ]; then
    echo "📦 Installing npm dependencies..."
    npm install
fi

npm run build

# Start services
echo "🌟 Starting backend server..."
cd ../backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &

# Serve frontend
echo "🌐 Starting frontend server..."
cd ../frontend
npm start &

echo "✅ SkinCare Pro is running!"
echo "📱 Frontend: http://localhost:3000"
echo "🔧 Backend API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"

# Keep script running
wait