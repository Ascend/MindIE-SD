# File and Directory Permission Requirements

MindIE SD APIs validate the security permissions of input files and directories. Common file and directory types and their permission requirements are listed below.

| File or directory | Permission requirement |
| -- | -- |
| Configuration files | The three permission groups must not exceed `640`, and they must be consistent with the group and permission expectations of the executing user. |
| Model weight files | The three permission groups must not exceed `640`, and they must be consistent with the group and permission expectations of the executing user. |
| Model weight directories | The three permission groups must not exceed `750`, and they must be consistent with the group and permission expectations of the executing user. |
