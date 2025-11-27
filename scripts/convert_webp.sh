find . -iname '*.webp' -exec bash -c 'ffmpeg -y -i "$1" "${1%.*}.jpg"' _ {} \;
