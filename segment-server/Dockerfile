FROM nikolaik/python-nodejs:latest


# Create app directory
WORKDIR /usr/src/app

# Copy both package.json (for Node.js) and requirements.txt (for Python)
COPY package*.json ./
COPY requirements.txt ./

# Install both npm and pip packages
RUN npm install
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application's source files
COPY . .

CMD [ "node", "app.js" ]
