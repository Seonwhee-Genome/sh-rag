FROM golang:1.23-alpine AS build
WORKDIR /src
# stdlib only: no module download step needed
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/gateway ./cmd/gateway

FROM alpine:3.20
RUN adduser -D -u 10001 app
USER app
COPY --from=build /out/gateway /usr/local/bin/gateway
EXPOSE 8080
HEALTHCHECK --interval=15s --timeout=3s --retries=3 CMD wget -qO- http://127.0.0.1:8080/healthz >/dev/null || exit 1
ENTRYPOINT ["/usr/local/bin/gateway"]
