#!/bin/bash
# Generate version header from git tags

VERSION_FILE="include/sparseflow/version.h"

# Get version from git
if git describe --tags --exact-match 2>/dev/null; then
    VERSION=$(git describe --tags --exact-match)
else
    VERSION="3.0.0-alpha+$(git rev-parse --short HEAD)"
fi

# Parse version
MAJOR=$(echo $VERSION | cut -d. -f1 | sed 's/v//')
MINOR=$(echo $VERSION | cut -d. -f2)
PATCH=$(echo $VERSION | cut -d. -f3 | cut -d- -f1 | cut -d+ -f1)

cat > $VERSION_FILE << EOH
// Auto-generated version header
#ifndef SPARSEFLOW_VERSION_H
#define SPARSEFLOW_VERSION_H

#define SPARSEFLOW_VERSION_MAJOR $MAJOR
#define SPARSEFLOW_VERSION_MINOR $MINOR
#define SPARSEFLOW_VERSION_PATCH $PATCH
#define SPARSEFLOW_VERSION_STRING "$VERSION"

#endif // SPARSEFLOW_VERSION_H
EOH

echo "✅ Generated $VERSION_FILE"
echo "   Version: $VERSION"
