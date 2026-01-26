# Base image: NVIDIA CUDA 12.1 with Ubuntu 22.04 (matches ROS 2 Humble)
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Avoid interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg2 \
    lsb-release \
    locales \
    python3 \
    python3-pip \
    python3-dev \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

# Add ROS 2 Humble repository
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2.list

# Install ROS 2 Humble base packages (minimal, no GUI)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-base \
    ros-humble-rosbag2 \
    ros-humble-rosbag2-storage-mcap \
    ros-humble-rosidl-runtime-py \
    ros-humble-sensor-msgs \
    ros-humble-sensor-msgs-py \
    ros-humble-geometry-msgs \
    ros-humble-std-msgs \
    ros-humble-nav-msgs \
    ros-humble-cv-bridge \
    && rm -rf /var/lib/apt/lists/*

# Set up ROS 2 environment
ENV ROS_DISTRO=humble
ENV AMENT_PREFIX_PATH=/opt/ros/humble
ENV CMAKE_PREFIX_PATH=/opt/ros/humble
ENV COLCON_PREFIX_PATH=/opt/ros/humble
ENV LD_LIBRARY_PATH=/opt/ros/humble/lib
ENV PATH=/opt/ros/humble/bin:$PATH
ENV PYTHONPATH=/opt/ros/humble/lib/python3.10/site-packages:/opt/ros/humble/local/lib/python3.10/dist-packages
ENV ROS_PYTHON_VERSION=3
ENV ROS_VERSION=2

WORKDIR /app

# Install Python dependencies
COPY requirements-docker.txt .
RUN pip3 install --no-cache-dir -r requirements-docker.txt

# Copy application code
COPY ./src /app/src
COPY ./api /app/api

EXPOSE 5000

ENV FLASK_APP=api/api.py
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Use gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "300", "api.api:app"]
