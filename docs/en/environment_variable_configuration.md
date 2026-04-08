# Environment Variable Configuration

After installation, the installation path provides a process-level environment setup script named `set_env.sh`. This script automatically exports the runtime environment variables required by MindIE SD. The variables shown in [Table 1](#table-environment-variables) are effective only for the current process and expire when the process exits.

**Table 1. Environment variables** <a id="table-environment-variables"></a>

| Environment variable | Description |
| -- | -- |
| `LD_LIBRARY_PATH` | Search path for dynamic libraries. |
| `ASCEND_CUSTOM_OPP_PATH` | Installation path of the custom operator package used by the inference engine. |
| `ASCEND_RT_VISIBLE_DEVICES` | Logical IDs of the Ascend AI processors visible to the current process. Examples: `"0,1,2"` or `"0-2"`. Use `,` to separate non-consecutive IDs and `-` for consecutive ranges. |
