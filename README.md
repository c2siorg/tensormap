<<<<<<< HEAD
=======
Here’s an updated version of your `README.md` file with Docker commands added to help users quickly set up and run TensorMap using Docker:

>>>>>>> e3fe572 (fix : fixing docker server compose file and adding dockerfile for client)
---

[![Build Status](https://travis-ci.com/scorelab/TensorMap.svg?branch=master)](https://travis-ci.com/scorelab/TensorMap)  
[![Join the chat at https://gitter.im/scorelab/TensorMap](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/scorelab/TensorMap)  
[![HitCount](http://hits.dwyl.com/scorelab/TensorMap.svg)](http://hits.dwyl.com/scorelab/TensorMap)

# TensorMap

TensorMap is a web application that allows users to create machine learning algorithms visually. TensorMap supports reverse engineering of the visual layout to a TensorFlow implementation in preferred languages. The goal of the project is to let beginners play with machine learning algorithms in TensorFlow without requiring extensive background knowledge about the library. For more details about the project, read our [project wiki](https://github.com/scorelab/TensorMap/wiki).

---

## Getting Started

Follow these steps to set up and run TensorMap using Docker.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)

---

### Running TensorMap with Docker

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/c2siorg/tensormap.git
   cd TensorMap
   ```

2. **Set Up Environment**:
   - Ensure Docker and Docker Compose are installed and running on your machine.

3. **Build and Run the Application**:
   Use Docker Compose to build and start the TensorMap services (database, server, and client):
   ```bash
   docker-compose up --build
   ```

   This will:
   - Start a PostgreSQL database.
   - Build and run the TensorMap server (Flask backend).
   - Build and run the TensorMap client (React frontend).

4. **Access the Application**:
   - **Frontend (Client)**: Open your browser and go to `http://localhost:5173`.
   - **Backend (Server)**: The Flask API will be available at `http://localhost:5000`.

5. **Stop the Application**:
   To stop the running services, press `Ctrl+C` in the terminal or run:
   ```bash
   docker-compose down
   ```

---

### Docker Compose Configuration
The `docker-compose.yml` file defines the following services:
- **Database**: PostgreSQL database for storing application data.
- **Server**: Flask backend for TensorMap.
- **Client**: React frontend for TensorMap.

You can modify the `docker-compose.yml` file to customize the setup (e.g., change ports or environment variables).

---

### Development with Docker
If you're developing TensorMap, you can use Docker to streamline your workflow:
- **Rebuild and Restart the Client**:
  ```bash
  docker-compose up --build client
  ```
- **View Logs**:
  ```bash
  docker-compose logs client
  ```
- **Access the Container Shell**:
  ```bash
  docker exec -it <client-container-id> /bin/sh
  ```

---

## Contributing

Please read the ['Note to Contributors'](https://github.com/scorelab/TensorMap/wiki/Note-to-Contributors) in the project wiki for more details.

---

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/scorelab/TensorMap/blob/master/LICENSE) file for details.

---

This updated `README.md` includes clear instructions for running TensorMap using Docker, making it easier for users to get started. Let me know if you need further adjustments!
