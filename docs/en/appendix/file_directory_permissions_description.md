# File and Directory Permissions

MindIE SD APIs perform permission security verification on the input files or folders. The following table describes common file and folder types and permission requirements.

|File|Permission Requirement|
|--|--|
|Config|The permissions of the three groups cannot exceed `640` and must be the same as the required groups and permission of the execution user.|
|Model weight file|The permissions of the three groups cannot exceed `640` and must be the same as the required groups and permission of the execution user.|
|Model weight folder|The three groups of permissions cannot exceed `750` and must be the same as the required groups and permissions of the execution user.|
