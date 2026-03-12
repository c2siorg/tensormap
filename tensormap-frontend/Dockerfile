FROM node:18-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json ./

RUN npm ci

COPY . .

<<<<<<< HEAD
ARG VITE_WS_URL
ENV VITE_WS_URL=$VITE_WS_URL
=======
ARG VITE_API_BASE_URL=http://localhost:4300/api/v1
ARG VITE_WS_URL=http://localhost:4300/dl-result

ENV VITE_API_BASE_URL=${VITE_API_BASE_URL:-http://localhost:4300/api/v1}
ENV VITE_WS_URL=${VITE_WS_URL}
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)

RUN npm run build


FROM nginx:alpine

COPY --from=builder /app/dist /usr/share/nginx/html

<<<<<<< HEAD
EXPOSE 80
=======
RUN echo 'server { \
  listen 80; \
  root /usr/share/nginx/html; \
  index index.html; \
  location / { try_files $uri $uri/ /index.html; } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 3300
>>>>>>> ac5c2d0 (fix: address review feedback for BatchNormalizationNode, tailwind config, jsconfig formatting, and add tests)

CMD ["nginx", "-g", "daemon off;"]
