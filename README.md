# Nemostation

This repo contains the Nemotron agent demo and supporting vendor libraries.

- `nemotron-agent-demo`: main demo, UI, and Docker Compose setup.
- `vendors`: optional vendor dependencies.

Start here:
```bash
cd nemotron-agent-demo
./ngc_login.sh
docker compose --env-file creds.env \
  -f docker-compose.yml \
  -f docker-compose.nemotron3-nim.yml up -d
```
