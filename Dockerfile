# Build stage
FROM golang:1.23.0-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

COPY . .

RUN CGO_ENABLED=0 GOOS=linux go build -o /app/server ./cmd/server

# Final stage
FROM alpine

WORKDIR /app

COPY --from=builder /app/server /app/server

USER 65532:65532

EXPOSE 8080

ENTRYPOINT ["/app/server"] 