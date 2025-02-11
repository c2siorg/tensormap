# Use an official Node.js runtime as the base image
FROM node:18-alpine

WORKDIR /app

COPY package.json package-lock.json* ./

RUN npm install

COPY . .

RUN npm run build
EXPOSE 5173

CMD ["npm", "start"]