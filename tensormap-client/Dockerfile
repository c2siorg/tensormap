# Use an official Node.js runtime as the base image
FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./

RUN npm install --legacy-peer-deps

COPY . .

EXPOSE 5173

CMD ["npm", "start"]