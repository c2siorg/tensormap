[![Build Status](https://travis-ci.com/scorelab/TensorMap.svg?branch=master)](https://travis-ci.com/scorelab/TensorMap)
[![Gitter Chat](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/scorelab/TensorMap)


# TensorMap - Visual Machine Learning Workflow Builder

**Version 2.1** 

---

## 🌟 Overview
TensorMap is an innovative web application that enables visual creation of machine learning algorithms with automatic translation to TensorFlow code. Designed to lower the barrier for ML beginners, it supports reverse engineering of visual layouts to production-ready implementations.

---

## 🚀 Key Features
- Drag-and-drop interface for neural network design
- Auto-generation of TensorFlow code (Python/JavaScript)
- Model visualization and version control
- Export capabilities for trained models
- Collaborative workspace support

---

## 🛠 Prerequisites
- Node.js v14+
- Python 3.9+
- TensorFlow 2.x
- PostgresSQL 4.4+
- npm v6+
- Docker (optional for containerized deployment)

---

## ⚡ Getting Started

### System Architecture
This repository has the following structure:
```
TensorMap/
├── tensormap-server/  # Backend services
├── tensormap-client/  # Frontend interface
├── docs/             # Documentation
└── scripts/          # Deployment 
```


### Installation

#### 1. Server Setup
```bash
cd tensormap-server
python -m venv env
pip install -r requirements.txt
cp .env.example .env  # Configure environment variables
python app.py
```
#### 2. Client Setup
```
cd tensormap-client
npm install
npm start
```

#### 3. Run in Development Mode
```
# From root directory
docker-compose -f docker-compose.dev.yml up --build
```
---

## 🧩 Core Components

### 1. Visual Builder
```
#Python
# Sample generated TensorFlow code

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])
```

### 2. Code Translator

+ Supports Python and JavaScript output
+ Custom layer configuration via JSON

```
// JSON

{
  "layer_type": "conv2d",
  "filters": 32,
  "kernel_size": [3, 3]
}
```
---
## 🔄 Development Workflow

### Branching Strategy

```
git checkout -b feat/new-layer-type   # Feature development
git checkout -b fix/issue-123         # Bug fixes
git checkout -b docs/readme-update    # Documentation improvements
```

### Testing

```
# Run server tests
cd tensormap-server
npm test

# Run client tests
cd tensormap-client
npm test
```

---
## 📈 Production Deployment

```
# Build production images
docker-compose -f docker-compose.prod.yml build

# Start services
docker-compose -f docker-compose.prod.yml up -d
```

### Environment Variables

```
#fill according your postgreSQL
db_name = 'database name'
db_host = 'host ip address' #comment this line out if the database is at localhost
db_user = 'database username'
db_password = 'database user password'
```

---
## 🤝 Contributing

#### Please read ['Note to Contributors'](https://github.com/scorelab/TensorMap/wiki/Note-to-Contributors) in [project wiki.](https://github.com/scorelab/TensorMap/wiki) for more details.

1. Fork the repository

1. Create your feature branch (git checkout -b feature/AmazingFeature)

1. Commit changes following Conventional Commits

1. Push to the branch (git push origin feature/AmazingFeature)

1. Open a Pull Request


### Code Standards

+ ES6+ for client code

+ Python PEP-8 for server code

+ 100% test coverage for new features


---
## 📚 Resources

[Project wiki.](https://github.com/scorelab/TensorMap/wiki)

---
## 📜 License

- This project is licensed under the MIT License
- see the [LICENSE.md](https://github.com/scorelab/TensorMap/blob/master/LICENSE) file for details

---
## 📞 Support
+ [GitHub Issues](https://github.com/c2siorg/tensormap/issues)


+ [Gitter Chat](	https://gitter.im/scorelab/TensorMap)

+ [Email Support](support@tensormap.org)

+ [Pull Requests](https://github.com/c2siorg/tensormap/pulls)
---
