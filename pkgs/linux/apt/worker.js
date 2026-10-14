// urna.dev/apt/*: the signed apt index from the urna-apt R2 bucket, and
// each package from the GitHub release that holds it. the bucket keeps only
// what aptsigner.yml writes (the keyring, Release, InRelease, Release.gpg
// and Packages per architecture); a .deb never lives here, so
// pool/<tag>/<file> answers a redirect to that release's asset, and apt
// checks the file against the SHA-256 the signed index names.

const REPO = "https://github.com/hoffresearch/urna/releases/download";
const POOL = /^pool\/(v\d+\.\d+\.\d+)\/(urna_\d+\.\d+\.\d+-\d+_(?:amd64|arm64)\.deb)$/;
const INDEX =
  /^(urna\.gpg|dists\/stable\/(?:InRelease|Release|Release\.gpg|main\/binary-(?:amd64|arm64)\/Packages(?:\.gz)?))$/;
const TYPES = {
  gpg: "application/pgp-keys",
  gz: "application/gzip",
};

export default {
  async fetch(request, env) {
    if (request.method !== "GET" && request.method !== "HEAD") {
      return new Response("method not allowed\n", { status: 405, headers: { Allow: "GET, HEAD" } });
    }
    const path = new URL(request.url).pathname.replace(/^\/apt\//, "");
    const pool = POOL.exec(path);
    if (pool) {
      return Response.redirect(`${REPO}/${pool[1]}/${pool[2]}`, 302);
    }
    if (!INDEX.test(path)) {
      return new Response("not found\n", { status: 404 });
    }
    const object = await env.APT.get(path);
    if (object === null) {
      return new Response("not found\n", { status: 404 });
    }
    const headers = new Headers();
    object.writeHttpMetadata(headers);
    headers.set("ETag", object.httpEtag);
    headers.set("Cache-Control", "public, max-age=300, no-transform");
    if (!headers.has("Content-Type")) {
      headers.set("Content-Type", TYPES[path.split(".").pop()] ?? "text/plain; charset=utf-8");
    }
    return new Response(request.method === "HEAD" ? null : object.body, { headers });
  },
};
