#!/bin/bash

# Copyright 2021 The KubeEdge Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -o errexit
set -o nounset
set -o pipefail

TMP_DIR=$(mktemp -d --suffix=.sedna)

trap "rm -rf '$TMP_DIR'" EXIT 

_download_yamls() {

  yaml_dir=$1
  mkdir -p ${TMP_DIR}/$yaml_dir
  cd ${TMP_DIR}/$yaml_dir
  for yaml in ${yaml_files[@]}; do
    echo downloading $yaml into ${TMP_DIR}/$yaml_dir
    curl -sSO https://raw.githubusercontent.com/kubeedge/sedna/main/$yaml_dir/$yaml
  done
}

download_yamls() {
  yaml_files=(
  sedna.io_datasets.yaml
  sedna.io_federatedlearningjobs.yaml
  sedna.io_incrementallearningjobs.yaml
  sedna.io_jointinferenceservices.yaml
  sedna.io_models.yaml
  )
  _download_yamls build/crds
  yaml_files=(
    gm.yaml
  )
  _download_yamls build/gm/rbac
}

prepare() {
  # we only need build directory
  # here don't use git clone because of large vendor directory
  download_yamls
}

prepare_gm_config_map() {
  cm_name=${1:-gm-config}
  config_file=${TMP_DIR}/${2:-gm.yaml}

  if [ -n "${SEDNA_GM_CONFIG:-}" ] && [ -f "${SEDNA_GM_CONFIG}" ] ; then
    cp "$SEDNA_GM_CONFIG" $config_file
  else
    cat > $config_file << EOF
kubeConfig: ""
master: ""
namespace: ""
websocket:
  address: 0.0.0.0
  port: 9000
localController:
  server: http://localhost:${SEDNA_LC_BIND_PORT:-9100}
EOF
  fi

  kubectl $action -n sedna configmap $cm_name --from-file=$config_file
}

do_gm() {
  cd ${TMP_DIR}

  kubectl $action -f build/crds

  kubectl $action -f build/gm/rbac/

  if [ "$action" == delete ] ; then 
    kubectl label node/$GM_NODE_NAME sedna- | sed 's/labeled$/un&/' || true
    return
  fi

  kubectl label node/$GM_NODE_NAME sedna=gm || true

  cm_name=gm-config
  config_file_name=gm.yaml
  prepare_gm_config_map $cm_name $config_file_name


  kubectl $action -f - <<EOF
apiVersion: v1
kind: Service
metadata:
  name: gm
  namespace: sedna
spec:
  selector:
    sedna: gm
  type: NodePort
  ports:
    - protocol: TCP
      port: 9000
      targetPort: 9000
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gm
  labels:
    sedna: gm
  namespace: sedna
spec:
  replicas: 1
  selector:
    matchLabels:
      sedna: gm
  template:
    metadata:
      labels:
        sedna: gm
    spec:
      nodeSelector:
        sedna: gm
      serviceAccountName: sedna
      containers:
      - name: gm
        image: kubeedge/sedna-gm:v0.1.0
        command: ["sedna-gm", "--config", "/config/$config_file_name", "-v2"]
        volumeMounts:
        - name: gm-config
          mountPath: /config
        resources:
          requests:
            memory: 32Mi
            cpu: 100m
          limits:
            memory: 128Mi
      volumes:
        - name: gm-config
          configMap:
            name: $cm_name
EOF
}

do_lc() {

  if [ "$action" == delete ] ; then 
    # ns would be deleted in do_gm
    # so no need to clean lc alone
    return
  fi

  gm_node_port=$(kubectl -n sedna get svc gm -ojsonpath='{.spec.ports[0].nodePort}')

  # here try to get node ip by kubectl
  gm_node_ip=$(kubectl get node $GM_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="ExternalIP")].address }')
  gm_node_internal_ip=$(kubectl get node $GM_NODE_NAME -o jsonpath='{ .status.addresses[?(@.type=="InternalIP")].address }')

  GM_ADDRESS=${gm_node_ip:-$gm_node_internal_ip}:$gm_node_port

  kubectl $action -f- <<EOF
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    sedna: lc
  name: lc
  namespace: sedna
spec:
  selector:
    matchLabels:
      sedna: lc
  template:
    metadata:
      labels:
        sedna: lc
    spec:
      containers:
        - name: lc
          image: kubeedge/sedna-lc:v0.1.0
          env:
            - name: GM_ADDRESS
              value: $GM_ADDRESS
            - name: BIND_PORT
              value: "${LC_BIND_PORT:-9100}"
            - name: NODENAME
              valueFrom:
                fieldRef:
                  fieldPath: spec.nodeName
            - name: ROOTFS_MOUNT_DIR
              # the value of ROOTFS_MOUNT_DIR is same with the mount path of volume
              value: /rootfs
          resources:
            requests:
              memory: 32Mi
              cpu: 100m
            limits:
              memory: 128Mi
          volumeMounts:
            - name: localcontroller
              mountPath: /rootfs
      volumes:
        - name: localcontroller
          hostPath:
            path: /
      restartPolicy: Always
      hostNetwork: true
EOF
}

wait_ok() {
  kubectl -n sedna wait --for=condition=available --timeout=600s deployment/gm
  kubectl -n sedna wait pod --for=condition=Ready --selector=sedna
  kubectl -n sedna get pod
}

check_kubectl () {
  kubectl get pod >/dev/null
}

check_action() {
  action=${SEDNA_ACTION:-create}
  support_action_list="create delete"
  if ! echo "$support_action_list" | grep -w -q "$action"; then
    echo "\`$action\` not in support action list: create/delete!" >&2
    echo "You need to specify it by setting $(red_text SEDNA_ACTION) environment variable when running this script!" >&2
    exit 2
  fi
  
}

check_gm_node() {
  GM_NODE_NAME=${SEDNA_GM_NODE:-}

  if [ -z "$GM_NODE_NAME" ] || ! kubectl get node $GM_NODE_NAME; then 
    echo "ERROR: $(red_text GM node name \`$GM_NODE_NAME\` does not exist in k8s cluster)!" >&2
    echo "You need to specify it by setting $(red_text SEDNA_GM_NODE) environment variable when running this script!" >&2
    exit 1
  fi
}

do_check() {
  check_kubectl
  check_action
  check_gm_node
}

show_debug_infos() {
  cat - <<EOF
Sedna is $(green_text running):
See GM status: kubectl -n sedna get deploy
See LC status: kubectl -n sedna get ds lc
See Pod status: kubectl -n sedna get pod
EOF
}

NO_COLOR='\033[0m'
RED='\033[0;31m'
GREEN='\033[0;32m'
green_text() {
  echo -ne "$GREEN$@$NO_COLOR"
}

red_text() {
  echo -ne "$RED$@$NO_COLOR"
}

do_check

prepare

do_gm

do_lc

case "$action" in
  create)
    wait_ok
    show_debug_infos
    ;;
  delete)
    echo "$(green_text Sedna is uninstalled successfully)"
    ;;
esac
