# File and Directory Permission Requirements

MindIE SD API performs permission security validation on passed files and directories. Common file and directory types and their permission requirements are described below:

| File | Permission Requirements |
| -- | -- |
| Config files | The three permission groups must not exceed 640 and must be consistent with the executing user's required group and permissions. |
| Model weight files | The three permission groups must not exceed 640 and must be consistent with the executing user's required group and permissions. |
| Model weight directories | The three permission groups must not exceed 750 and must be consistent with the executing user's required group and permissions. |
