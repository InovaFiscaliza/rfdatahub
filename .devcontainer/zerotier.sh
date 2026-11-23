docker volume create devcontainer-zerotier

docker run --rm --name myzerotier -d \
  --network host \
  --cap-add NET_ADMIN \
  --device /dev/net/tun \
  --mount source=devcontainer-zerotier,target=/var/lib/zerotier-one \
  zerotier/zerotier:latest