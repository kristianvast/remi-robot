import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { $ } from "./display-math.js";
import { car3d, show3dFallback } from "./car3d.js";

export function loadCarModel(url) {
  if (!car3d.ok || car3d.modelRequested) return;
  car3d.modelRequested = true;
  $("car3d-status").textContent = "SYNCING GLB...";

  const fresnelMat = new THREE.ShaderMaterial({
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    uniforms: {
      uColor: { value: new THREE.Color(0x26e8ff) },
    },
    vertexShader: `
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        vec4 mvPos = modelViewMatrix * vec4(position, 1.0);
        vNormal = normalize(normalMatrix * normal);
        vViewDir = normalize(-mvPos.xyz);
        gl_Position = projectionMatrix * mvPos;
      }
    `,
    fragmentShader: `
      uniform vec3 uColor;
      varying vec3 vNormal;
      varying vec3 vViewDir;
      void main() {
        float fresnel = pow(1.0 - abs(dot(normalize(vNormal), normalize(vViewDir))), 2.6);
        gl_FragColor = vec4(uColor, fresnel * 0.42);
      }
    `,
  });

  const loader = new GLTFLoader();
  loader.load(
    url,
    (gltf) => {
      const model = gltf.scene;
      // Rotate -90 deg about X only if the GLB is still raw URDF Z-up (not already converted to glTF Y-up)
      const root = new THREE.Group();
      root.add(model);
      root.updateMatrixWorld(true);
      const baseNode = model.getObjectByName("base_link");
      const panNode = model.getObjectByName("sensor_pan");
      if (baseNode && panNode) {
        const bp = new THREE.Vector3();
        const pp = new THREE.Vector3();
        baseNode.getWorldPosition(bp);
        panNode.getWorldPosition(pp);
        if (Math.abs(pp.z - bp.z) > Math.abs(pp.y - bp.y)) {
          model.rotation.x = -Math.PI / 2;
        }
      } else {
        model.rotation.x = -Math.PI / 2;
      }

      // Apply holographic styling: dimmed deep-tech base + additive cyan edges + subtle fresnel rim
      const meshes = [];
      model.traverse((obj) => {
        if (obj.isMesh) meshes.push(obj);
      });
      const edgeMat = new THREE.LineBasicMaterial({
        color: 0x3cf0ff,
        transparent: true,
        opacity: 0.56,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      for (const m of meshes) {
        const isWheel = /wheel/i.test(m.name);
        m.material = new THREE.MeshStandardMaterial({
          color: isWheel ? 0x0b1f2e : 0x0d2b40,
          emissive: isWheel ? 0x052436 : 0x083852,
          emissiveIntensity: 0.85,
          metalness: 0.55,
          roughness: 0.38,
          transparent: true,
          opacity: 0.86,
        });
        if (m.geometry) {
          const edges = new THREE.EdgesGeometry(m.geometry, 26);
          const line = new THREE.LineSegments(edges, edgeMat);
          m.add(line);
          const rim = new THREE.Mesh(m.geometry, fresnelMat);
          m.add(rim);
        }
      }

      // Ground & center on origin
      root.updateMatrixWorld(true);
      const box = new THREE.Box3().setFromObject(root);
      const center = box.getCenter(new THREE.Vector3());
      root.position.set(-center.x, -box.min.y, -center.z);
      car3d.scene.add(root);
      car3d.root = root;

      // Bind named URDF nodes
      car3d.wheels.fl = model.getObjectByName("front_left_wheel");
      car3d.wheels.fr = model.getObjectByName("front_right_wheel");
      car3d.wheels.rl = model.getObjectByName("rear_left_wheel");
      car3d.wheels.rr = model.getObjectByName("rear_right_wheel");
      car3d.sensorPan = model.getObjectByName("sensor_pan");
      if (car3d.sensorPan) car3d.sensorPanBaseZ = car3d.sensorPan.rotation.z;

      // Translucent sonar cone attached to ultrasonic_link extending along +X (URDF forward)
      const ultra = model.getObjectByName("ultrasonic_link") || car3d.sensorPan || model;
      const coneGeo = new THREE.ConeGeometry(1, 1, 28, 1, true);
      coneGeo.translate(0, -0.5, 0);        // apex at origin, base at y=-1
      coneGeo.rotateZ(Math.PI / 2);         // base along +X (URDF forward)
      const coneMat = new THREE.MeshBasicMaterial({
        color: 0x26e8ff,
        transparent: true,
        opacity: 0.24,
        side: THREE.DoubleSide,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      });
      const coneMesh = new THREE.Mesh(coneGeo, coneMat);
      coneMesh.scale.set(0.35, 0.06, 0.06);
      ultra.add(coneMesh);
      car3d.sonarCone = coneMesh;
      car3d.sonarMat = coneMat;

      // Frame camera cleanly to car bounding sphere so wheels, chassis, and sonar origin sit inside with margin
      const size = box.getSize(new THREE.Vector3());
      const sphere = box.getBoundingSphere(new THREE.Sphere());
      car3d.fitRadius = Math.max(0.15, sphere.radius);
      car3d.targetY = size.y * 0.48;
      car3d.orbitDist = car3d.fitRadius * 3.7;

      window.__remi.carLoaded = true;
      $("car3d-status").textContent = "MODEL LOCKED";
    },
    undefined,
    (err) => {
      show3dFallback(`3D chassis load failed (${url}): ${err && err.message ? err.message : "network error"}`);
    }
  );
}
