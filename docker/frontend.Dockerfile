FROM node:22.12.0-alpine3.21 AS build

WORKDIR /app

COPY TKBEN/client/package.json TKBEN/client/package-lock.json ./
RUN npm ci

COPY TKBEN/client ./
RUN npm run build

FROM nginx:1.27.5-alpine3.21

COPY docker/nginx/default.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
