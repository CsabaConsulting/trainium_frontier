/* Strip provider credentials before Claude Code's sandboxed shell runs code. */
#define _GNU_SOURCE
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(int argc, char **argv) {
    (void)argc;
    /* The outer bubblewrap owns filesystem/PID isolation. Nested unshare is
       prohibited on this workstation. Remove provider credentials from the
       command environment; no host SSH/AWS files are mounted in the sandbox. */
    unsetenv("AWS_BEARER_TOKEN_BEDROCK");
    unsetenv("AWS_ACCESS_KEY_ID");
    unsetenv("AWS_SECRET_ACCESS_KEY");
    unsetenv("AWS_SESSION_TOKEN");
    unsetenv("ANTHROPIC_API_KEY");
    unsetenv("OPENAI_API_KEY");
    argv[0] = "/opt/bash_real";
    execv(argv[0], argv);
    fprintf(stderr, "credential-free shell exec failed: %d\n", errno);
    return 127;
}
