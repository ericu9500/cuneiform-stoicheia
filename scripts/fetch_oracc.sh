#!/bin/bash
# Download every public Oracc project as its JSON zip (kept zipped). TLS is verified with the system
# bundle plus the InCommon RSA Server CA 2 intermediate that the Oracc server omits (data/raw/oracc-ca.pem).
cd "$(dirname "$0")/../data/raw/oracc"
python3 -c "import json;[print(p) for p in json.load(open('projects.json'))['public']]" | while read p; do
  f=$(echo "$p" | tr / -).zip
  [ -s "$f" ] && continue
  curl -sS --fail -m 1800 --cacert ../oracc-ca.pem -o "$f.part" -w "$p %{http_code} %{size_download}\n" "https://oracc.museum.upenn.edu/json/$f" && mv "$f.part" "$f"
  sleep 1
done
