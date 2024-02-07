# Use the official Ubuntu 20.04 base image
FROM ubuntu:20.04

# Set non-interactive mode
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Expose the Streamlit default port
EXPOSE 8501

# Copy the Streamlit app code into the container
COPY . .

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
