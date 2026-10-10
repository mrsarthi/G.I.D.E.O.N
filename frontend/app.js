// ═══════════════════════════════════════════════════════════════
//  G.I.D.E.O.N — Holographic AI Interface
//  Loads gideon_hologram.glb with holographic shader treatment
// ═══════════════════════════════════════════════════════════════
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

/* ───────────────────────────────────────────────────────────────
 *  GLSL SHADERS — Holographic Surface Material
 * ─────────────────────────────────────────────────────────────── */
const HOLO_VERT = `
varying vec3 vNormal;
varying vec3 vPos;
varying vec3 vWorldPos;

uniform float uAudioVolume;

void main() {
    vNormal   = normalize(normalMatrix * normal);
    vPos      = position;
    
    vec3 localPos = position;
    
    // Weight function for jaw/mouth area
    // Max coordinates: [97, 95, 156], Min coordinates: [-97, -149, -124]
    // Mouth/jaw is situated in lower front:
    // Y between -100.0 and -20.0
    // Z (front face) is positive (> 15.0)
    // X (width) is centered (fade out towards ears at abs(X) > 80)
    float weightY = smoothstep(-100.0, -70.0, localPos.y) * (1.0 - smoothstep(-45.0, -20.0, localPos.y));
    float weightZ = smoothstep(15.0, 70.0, localPos.z);
    float weightX = 1.0 - smoothstep(10.0, 80.0, abs(localPos.x));
    float weight = weightY * weightZ * weightX;
    
    // Deform downwards on Y and slightly forwards on Z
    localPos.y -= weight * uAudioVolume * 12.0;
    localPos.z += weight * uAudioVolume * 4.0;
    
    vWorldPos = (modelMatrix * vec4(localPos, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(localPos, 1.0);
}`;

const HOLO_FRAG = `
uniform float uTime;
uniform vec3  uColor;
uniform vec3  uAccent;
uniform float uIntensity;

varying vec3 vNormal;
varying vec3 vPos;
varying vec3 vWorldPos;

float hash(float n) { return fract(sin(n) * 43758.5453); }

void main() {
    vec3 viewDir = normalize(cameraPosition - vWorldPos);

    // ── Fresnel edge glow ──
    float fresnel = pow(1.0 - abs(dot(vNormal, viewDir)), 2.8);

    // ── Horizontal scan lines ──
    float scan = sin(vPos.y * 80.0 - uTime * 2.0) * 0.5 + 0.5;
    scan = pow(scan, 14.0) * 0.35;

    // ── Sweeping scan bar ──
    float barY = mod(uTime * 0.45, 4.0) - 2.0;
    float bar  = 1.0 - smoothstep(0.0, 0.1, abs(vPos.y - barY));
    bar *= 0.55;

    // ── Micro data grid ──
    float gx   = smoothstep(0.97, 1.0, abs(sin(vPos.x * 28.0)));
    float gy   = smoothstep(0.97, 1.0, abs(sin(vPos.y * 28.0)));
    float grid = max(gx, gy) * 0.06;

    // ── Glitch noise ──
    float n     = hash(floor(uTime * 18.0) + floor(vPos.y * 7.0));
    float glitch = n > 0.97 ? 0.35 : 0.0;

    // ── Alpha composite ──
    float alpha = fresnel * 0.6 + scan + bar + grid + 0.04 - glitch;
    alpha = clamp(alpha, 0.0, 1.0) * uIntensity;

    // ── Color ──
    vec3 col = uColor * (0.35 + fresnel * 0.85 + scan * 0.4);
    col += uAccent * bar * 0.4;
    col += vec3(0.8, 1.0, 0.95) * bar * 0.15;

    gl_FragColor = vec4(col, alpha);
}`;



/* ═══════════════════════════════════════════════════════════════
 *  MAIN APPLICATION
 * ═══════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', () => {

    // ─── UI Element References ───
    const clockEl           = document.getElementById('clock');
    const alertPopup        = document.getElementById('alert-popup');
    const leftSidebar       = document.getElementById('left-sidebar');
    const rightSidebar      = document.getElementById('right-sidebar');
    const hologramContainer = document.getElementById('hologram-container');
    const avatarZone        = document.getElementById('avatar-zone');
    const chatInput         = document.getElementById('chat-input');
    const sendBtn           = document.getElementById('send-btn');
    const micBtn            = document.getElementById('mic-btn');
    const promptChips       = document.querySelectorAll('.prompt-chip');
    const cpuVal            = document.getElementById('cpu-val');
    const cpuBar            = document.getElementById('cpu-bar');
    const gpuVal            = document.getElementById('gpu-val');
    const gpuBar            = document.getElementById('gpu-bar');
    const latencyVal        = document.getElementById('latency-val');
    const latencyBar        = document.getElementById('latency-bar');
    const ramVal            = document.getElementById('ram-val');
    const navDashboard      = document.getElementById('nav-dashboard');
    const navNeural         = document.getElementById('nav-neural');
    const navArchive        = document.getElementById('nav-archive');

    let targetRotX = 0;
    let targetRotY = 0;
    let currentSessionId = null;
    let activeTypewriter = null;

    // ─── Audio Engine, FIFO Queue & Visualizer for Lip Sync ───
    let audioContext = null;
    let analyser = null;
    let currentAudio = null;
    let audioSource = null;
    let uAudioVolumeValue = 0.0;
    let audioDataArray = null;
    
    let audioQueue = [];
    let isPlayingAudio = false;

    function stopAndClearAudio() {
        audioQueue = [];
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.src = "";
        }
        isPlayingAudio = false;
    }

    function enqueueAudio(url) {
        audioQueue.push(url);
        processAudioQueue();
    }

    function processAudioQueue() {
        if (isPlayingAudio || audioQueue.length === 0) return;

        isPlayingAudio = true;
        const url = audioQueue.shift();

        try {
            if (currentAudio) {
                currentAudio.pause();
                currentAudio.src = "";
            }

            if (!audioContext) {
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                analyser.fftSize = 256;
                audioDataArray = new Uint8Array(analyser.frequencyBinCount);
            }

            if (audioContext.state === 'suspended') {
                audioContext.resume();
            }

            currentAudio = new Audio(url);
            currentAudio.crossOrigin = "anonymous";

            audioSource = audioContext.createMediaElementSource(currentAudio);
            audioSource.connect(analyser);
            analyser.connect(audioContext.destination);

            currentAudio.onended = () => {
                isPlayingAudio = false;
                processAudioQueue();
            };

            currentAudio.play().catch(e => {
                console.error("Audio playback failed:", e);
                isPlayingAudio = false;
                processAudioQueue();
            });
        } catch (err) {
            console.error("Failed to play audio chunk:", err);
            isPlayingAudio = false;
            processAudioQueue();
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  THREE.JS SCENE SETUP
    // ═══════════════════════════════════════════════════════════
    const container = document.getElementById('canvas-container');
    const W = container.clientWidth  || 600;
    const H = container.clientHeight || 600;

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 1000);
    camera.position.set(0, 0.1, 4.5);

    // low-power: on a dual-GPU laptop the browser renders on the integrated
    // GPU, leaving the GTX's VRAM and power budget to the model.
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: 'low-power' });
    renderer.setSize(W, H);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.4;
    container.appendChild(renderer.domElement);

    // Handle resize
    const resizeObserver = new ResizeObserver(entries => {
        for (const entry of entries) {
            const { width, height } = entry.contentRect;
            if (width > 0 && height > 0) {
                camera.aspect = width / height;
                camera.updateProjectionMatrix();
                renderer.setSize(width, height);
            }
        }
    });
    resizeObserver.observe(container);

    // ─── Lighting Rig ───
    scene.add(new THREE.AmbientLight(0x182040, 1.2));

    const keyLight = new THREE.DirectionalLight(0x5ffbd6, 3.5);
    keyLight.position.set(-2, 3, 3);
    scene.add(keyLight);

    const fillLight = new THREE.PointLight(0xc5a3ff, 2.0, 12);
    fillLight.position.set(3, -1, 2);
    scene.add(fillLight);

    const rimLight = new THREE.PointLight(0x5ffbd6, 1.8, 10);
    rimLight.position.set(0, 1, -3);
    scene.add(rimLight);

    // ═══════════════════════════════════════════════════════════
    //  HOLOGRAPHIC MODEL GROUP
    // ═══════════════════════════════════════════════════════════
    const headGroup = new THREE.Group();
    scene.add(headGroup);

    // Holographic shader material (applied to loaded model)
    const holoMat = new THREE.ShaderMaterial({
        uniforms: {
            uTime:      { value: 0 },
            uColor:     { value: new THREE.Color(0x5ffbd6) },
            uAccent:    { value: new THREE.Color(0xc5a3ff) },
            uIntensity: { value: 0.88 },
            uAudioVolume: { value: 0.0 }
        },
        vertexShader:   HOLO_VERT,
        fragmentShader: HOLO_FRAG,
        transparent: true,
        side:        THREE.DoubleSide,
        depthWrite:  false,
        blending:    THREE.AdditiveBlending
    });

    // Wireframe material for overlay (using the same shader so it deforms in sync)
    const wireMat = holoMat.clone();
    wireMat.wireframe = true;
    wireMat.uniforms.uIntensity.value = 0.07;

    // ─── Load gideon_hologram.glb ───
    const loader = new GLTFLoader();
    let modelLoaded = false;

    loader.load(
        'model/gideon_hologram.glb?v=' + Date.now(),
        (gltf) => {
            const model = gltf.scene;

            // Auto-center and auto-scale
            const box    = new THREE.Box3().setFromObject(model);
            const size   = box.getSize(new THREE.Vector3());
            const center = box.getCenter(new THREE.Vector3());
            const maxDim = Math.max(size.x, size.y, size.z) || 1;
            const scale  = 2.8 / maxDim;

            model.scale.setScalar(scale);
            model.position.set(
                -center.x * scale,
                -center.y * scale,
                -center.z * scale
            );
            model.rotation.x = 0.28; // Tilt head forward slightly more to look directly at the user

            // Apply holographic shader to all meshes
            model.traverse((child) => {
                if (child.isMesh) {
                    child.material = holoMat;
                }
            });

            headGroup.add(model);

            // Create wireframe clone
            const wireClone = model.clone(true);
            wireClone.traverse((child) => {
                if (child.isMesh) {
                    child.material = wireMat;
                }
            });
            wireClone.scale.setScalar(scale * 1.012);
            wireClone.position.copy(model.position);
            headGroup.add(wireClone);

            modelLoaded = true;
            triggerAlert("G.I.D.E.O.N: Holographic avatar loaded. Systems nominal.");
        },
        (xhr) => {
            if (xhr.total) {
                const pct = Math.round((xhr.loaded / xhr.total) * 100);
                if (pct % 25 === 0) console.log(`Loading model: ${pct}%`);
            }
        },
        (error) => {
            console.error('Model load failed:', error);
            triggerAlert("G.I.D.E.O.N: Avatar load failed. Fallback active.");

            // Fallback — glowing sphere
            const geo = new THREE.SphereGeometry(1.0, 48, 48);
            const mesh = new THREE.Mesh(geo, holoMat);
            headGroup.add(mesh);

            const wire = new THREE.Mesh(
                new THREE.SphereGeometry(1.0, 16, 16),
                wireMat
            );
            wire.scale.setScalar(1.015);
            headGroup.add(wire);

            modelLoaded = true;
        }
    );

    // ▸ Scan ring — horizontal torus that sweeps up/down
    const scanGeo = new THREE.TorusGeometry(1.4, 0.004, 8, 120);
    const scanMat = new THREE.MeshBasicMaterial({
        color: 0x5ffbd6, transparent: true, opacity: 0.50,
        blending: THREE.AdditiveBlending
    });
    const scanRing = new THREE.Mesh(scanGeo, scanMat);
    scanRing.rotation.x = Math.PI / 2;
    headGroup.add(scanRing);



    // ═══════════════════════════════════════════════════════════
    //  REACTIVE STATE
    // ═══════════════════════════════════════════════════════════
    let reactiveBoost = 0;

    function triggerReaction(duration = 2000) {
        reactiveBoost = 1.0;
        setTimeout(() => { reactiveBoost = 0; }, duration);
    }

    // ═══════════════════════════════════════════════════════════
    //  ANIMATION LOOP
    // ═══════════════════════════════════════════════════════════
    const threeClock = new THREE.Clock();

    // On battery the hologram drops to ~12 fps: this laptop powers off if the
    // GPU draws too much, and a 60 fps shader loop is pure decoration.
    let onBattery = false;
    let lastFrameAt = 0;
    const BATTERY_FRAME_SECONDS = 1 / 12;

    function animate() {
        requestAnimationFrame(animate);
        const t = threeClock.getElapsedTime();
        if (onBattery && t - lastFrameAt < BATTERY_FRAME_SECONDS) return;
        lastFrameAt = t;

        // Shader uniforms
        holoMat.uniforms.uTime.value = t;
        wireMat.uniforms.uTime.value = t;

        // Reactive intensity decay
        reactiveBoost *= 0.97;
        holoMat.uniforms.uIntensity.value = 0.88 + reactiveBoost * 0.4;

        // ── Head breathing / float ──
        headGroup.position.y = 0.45 + Math.sin(t * 1.2) * 0.025;

        // ── Smooth head rotation (natural idle look-around) ──
        targetRotY = Math.sin(t * 0.5) * 0.15 + Math.cos(t * 0.2) * 0.08;
        targetRotX = Math.cos(t * 0.4) * 0.08 + Math.sin(t * 0.15) * 0.04;
        headGroup.rotation.y += (targetRotY - headGroup.rotation.y) * 0.05;
        headGroup.rotation.x += (targetRotX - headGroup.rotation.x) * 0.05;

        // ── Wireframe pulse (now uIntensity uniform) ──
        wireMat.uniforms.uIntensity.value = 0.06 + Math.sin(t * 1.8) * 0.02;

        // ── Audio Analyser for Lip Sync ──
        if (analyser && currentAudio && !currentAudio.paused && !currentAudio.ended) {
            analyser.getByteFrequencyData(audioDataArray);
            let sum = 0;
            const len = Math.floor(audioDataArray.length * 0.6);
            for (let i = 0; i < len; i++) {
                sum += audioDataArray[i];
            }
            const average = sum / len;
            uAudioVolumeValue = Math.min(1.0, average / 130.0);
        } else {
            uAudioVolumeValue *= 0.85;
            if (uAudioVolumeValue < 0.001) uAudioVolumeValue = 0.0;
        }

        holoMat.uniforms.uAudioVolume.value = uAudioVolumeValue;
        wireMat.uniforms.uAudioVolume.value = uAudioVolumeValue;

        // ── Scan ring sweep ──
        const scanY = Math.sin(t * 0.7) * 1.5;
        scanRing.position.y = scanY;
        const sScale = Math.max(0.1, 1.0 - Math.abs(scanY) * 0.25);
        scanRing.scale.set(sScale, 1, sScale);
        scanMat.opacity = 0.45 * (1.0 - Math.abs(scanY) * 0.4);

        // Render
        renderer.render(scene, camera);
    }

    animate();

    // ═══════════════════════════════════════════════════════════
    //  UI — CLOCK
    // ═══════════════════════════════════════════════════════════
    function updateClock() {
        const now  = new Date();
        const hrs  = now.getHours().toString().padStart(2, '0');
        const mins = now.getMinutes().toString().padStart(2, '0');
        const secs = now.getSeconds().toString().padStart(2, '0');
        clockEl.textContent = `${hrs}:${mins}:${secs}`;
    }
    setInterval(updateClock, 1000);
    updateClock();

    // ═══════════════════════════════════════════════════════════
    //  UI — SIMULATED METRICS
    // ═══════════════════════════════════════════════════════════
    async function updateSystemMetrics() {
        try {
            const res = await fetch('/api/metrics');
            if (!res.ok) throw new Error('API response not OK');
            const data = await res.json();
            onBattery = !!data.on_battery;

            // CPU
            const cpu = Math.round(data.cpu);
            cpuVal.textContent = `${cpu}%`;
            cpuBar.style.width = `${cpu}%`;

            // GPU
            const gpu = Math.round(data.gpu);
            gpuVal.textContent = `${gpu}%`;
            gpuBar.style.width = `${gpu}%`;

            // Latency
            const latency = Math.round(data.latency);
            latencyVal.textContent = `${latency}ms`;
            latencyBar.style.width = `${Math.min(100, Math.floor((latency / 100) * 100))}%`;

            // RAM
            const ramUsed = data.ram_used.toFixed(1);
            const ramTotal = data.ram_total.toFixed(1);
            ramVal.innerHTML = `${ramUsed} GB <span class="text-sm opacity-30 font-data-mono">/ ${ramTotal}GB</span>`;

            // Flux Density
            const fluxVal = document.querySelector('#metrics-view div:nth-child(2) p.text-3xl');
            if (fluxVal) {
                fluxVal.innerHTML = `${data.flux.toFixed(1)}<span class="text-sm opacity-30 font-data-mono">MBPS</span>`;
            }

            // Drive Status
            const driveStatusEl = document.querySelector('#metrics-view div:nth-child(3) span');
            const driveDotEl = document.querySelector('#metrics-view div:nth-child(3) div');
            if (driveStatusEl) {
                driveStatusEl.textContent = `Drive: ${data.drive}`;
                if (data.drive === "Stable") {
                    driveStatusEl.className = "text-[9px] font-data-mono text-pulse-emerald/60 tracking-widest uppercase";
                    if (driveDotEl) {
                        driveDotEl.className = "w-1.5 h-1.5 rounded-full bg-pulse-emerald shadow-[0_0_8px_#00FF9D]";
                    }
                } else {
                    driveStatusEl.className = "text-[9px] font-data-mono text-system-red/80 tracking-widest uppercase animate-pulse";
                    if (driveDotEl) {
                        driveDotEl.className = "w-1.5 h-1.5 rounded-full bg-system-red shadow-[0_0_8px_#FF2D55] animate-pulse";
                    }
                }
            }
        } catch (err) {
            console.warn('Failed to fetch real-time system metrics, falling back to simulation:', err);
            const cpu = Math.floor(Math.random() * (55 - 25 + 1)) + 25;
            cpuVal.textContent = `${cpu}%`;
            cpuBar.style.width = `${cpu}%`;

            const gpu = Math.floor(Math.random() * (28 - 8 + 1)) + 8;
            gpuVal.textContent = `${gpu}%`;
            gpuBar.style.width = `${gpu}%`;

            const latency = Math.floor(Math.random() * (22 - 11 + 1)) + 11;
            latencyVal.textContent = `${latency}ms`;
            latencyBar.style.width = `${Math.floor((latency / 30) * 100)}%`;

            const ram = (Math.random() * (6.5 - 4.2) + 4.2).toFixed(1);
            ramVal.innerHTML = `${ram} GB <span class="text-sm opacity-30 font-data-mono">/ 16.0GB</span>`;
        }
    }
    setInterval(updateSystemMetrics, 3000);
    updateSystemMetrics();

    // ═══════════════════════════════════════════════════════════
    //  UI — MOUSE PARALLAX & ROTATION
    // ═══════════════════════════════════════════════════════════
    document.addEventListener('mousemove', (e) => {
        const x = (e.clientX - window.innerWidth  / 2) / 100;
        const y = (e.clientY - window.innerHeight / 2) / 100;

        leftSidebar.style.transform       = `translate(${x * 1.5}px, calc(-50% + ${y * 1.5}px))`;
        rightSidebar.style.transform      = `translate(${x * -1.5}px, calc(-50% + ${y * -1.5}px))`;
        hologramContainer.style.transform = `translate(${x * 3}px, ${y * 3}px)`;
    });

    // ═══════════════════════════════════════════════════════════
    //  UI — ALERT SYSTEM
    // ═══════════════════════════════════════════════════════════
    let alertTimeout;
    function triggerAlert(message) {
        alertPopup.textContent = message || "G.I.D.E.O.N: Interface active.";
        alertPopup.classList.add('show');
        clearTimeout(alertTimeout);
        alertTimeout = setTimeout(() => {
            alertPopup.classList.remove('show');
        }, 3000);
    }

    // Initial alert
    triggerAlert("G.I.D.E.O.N: Loading holographic avatar...");

    // ═══════════════════════════════════════════════════════════
    //  UI — CHAT API INTEGRATION & SESSION MANAGEMENT
    // ═══════════════════════════════════════════════════════════
    async function loadSessions(selectAfterLoadId = null) {
        try {
            const res = await fetch('/api/sessions');
            const sessions = await res.json();
            const selectEl = document.getElementById('session-select');
            const selectHeaderEl = document.getElementById('session-select-header');
            
            const selects = [selectEl, selectHeaderEl].filter(Boolean);
            if (selects.length === 0) return;

            selects.forEach(sel => {
                sel.innerHTML = '';
            });
            
            if (sessions.length === 0) {
                const newRes = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: '' })
                });
                const defaultSession = await newRes.json();
                currentSessionId = defaultSession.session_id;
                
                selects.forEach(sel => {
                    const opt = document.createElement('option');
                    opt.value = defaultSession.session_id;
                    opt.textContent = defaultSession.session_name;
                    opt.className = "bg-surface text-on-surface text-xs";
                    sel.appendChild(opt);
                });
            } else {
                sessions.forEach(s => {
                    selects.forEach(sel => {
                        const opt = document.createElement('option');
                        opt.value = s.session_id;
                        opt.textContent = s.session_name;
                        opt.className = "bg-surface text-on-surface text-xs";
                        sel.appendChild(opt);
                    });
                });
                
                if (selectAfterLoadId) {
                    currentSessionId = selectAfterLoadId;
                } else if (!currentSessionId) {
                    currentSessionId = sessions[0].session_id;
                }
                
                selects.forEach(sel => {
                    sel.value = currentSessionId;
                });
            }
            
            loadHistory(currentSessionId);
        } catch (err) {
            console.error('Failed to load G.I.D.E.O.N sessions:', err);
            triggerAlert("G.I.D.E.O.N: Session link error. Server down?");
        }
    }

    async function loadHistory(sessionId) {
        if (!sessionId) return;
        try {
            const res = await fetch(`/api/sessions/${sessionId}/history`);
            const messages = await res.json();
            
            const logEl = document.getElementById('chat-history-log');
            if (!logEl) return;
            logEl.innerHTML = '';
            
            messages.forEach(msg => {
                const bubble = document.createElement('div');
                bubble.className = `chat-bubble chat-bubble-${msg.role}`;
                
                const content = document.createElement('div');
                content.className = 'chat-bubble-content';
                if (msg.role === 'tool') {
                    content.className += ' chat-bubble-tool';
                }
                content.textContent = msg.content;
                bubble.appendChild(content);
                
                const meta = document.createElement('div');
                meta.className = 'chat-bubble-meta';
                
                let timeStr = '';
                if (msg.timestamp) {
                    const dt = new Date(msg.timestamp);
                    timeStr = dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
                }
                meta.textContent = `${msg.role} | ${timeStr}`;
                bubble.appendChild(meta);
                
                logEl.appendChild(bubble);
            });
            
            const historyView = document.getElementById('history-view');
            if (historyView) {
                historyView.scrollTop = historyView.scrollHeight;
            }
            loadPendingApprovals(sessionId);
        } catch (err) {
            console.error('Failed to load message history:', err);
        }
    }

    function runTypewriter(text, elementId, speed = 15) {
        const el = document.getElementById(elementId);
        if (!el) return;
        
        if (activeTypewriter) {
            clearInterval(activeTypewriter);
        }
        
        el.textContent = '';
        let index = 0;
        
        const dot = document.querySelector('.hud-dot');
        if (dot) dot.classList.add('pulsing');
        
        activeTypewriter = setInterval(() => {
            if (index < text.length) {
                el.textContent += text.charAt(index);
                index++;
                const hudBody = el.parentElement;
                if (hudBody) hudBody.scrollTop = hudBody.scrollHeight;
            } else {
                clearInterval(activeTypewriter);
                activeTypewriter = null;
                if (dot) dot.classList.remove('pulsing');
            }
        }, speed);
    }

    // ═══════════════════════════════════════════════════════════
    //  UI — APPROVALS
    //  Privileged actions wait here until the user clicks. The model
    //  cannot approve its own request; only this button can.
    // ═══════════════════════════════════════════════════════════
    const approvalStack = document.getElementById('approval-stack');
    const approvalCards = new Map();   // action_id -> card element

    const STATUS_TITLES = {
        pending: 'Approval needed',
        approved: 'Approved — done',
        failed: 'Approved — failed',
        denied: 'Denied',
        expired: 'Expired',
    };

    function authHeaders() {
        return { 'Content-Type': 'application/json', 'X-Gideon-Token': window.GIDEON_API_TOKEN || '' };
    }

    function renderApproval(action) {
        if (!approvalStack || !action || action.action_id == null) return;

        let card = approvalCards.get(action.action_id);
        if (!card) {
            card = document.createElement('div');
            card.className = 'approval-card';
            approvalStack.appendChild(card);
            approvalCards.set(action.action_id, card);
        }
        // Built with textContent only: the summary contains model-chosen
        // arguments, which must never be interpreted as markup.
        card.innerHTML = '';
        card.className = 'approval-card' + (action.status === 'pending' ? '' : ` resolved ${action.status}`);

        const title = document.createElement('div');
        title.className = 'approval-title';
        title.textContent = `${STATUS_TITLES[action.status] || action.status} · #${action.action_id}`;
        card.appendChild(title);

        const summary = document.createElement('div');
        summary.className = 'approval-summary';
        summary.textContent = action.summary || action.tool;
        card.appendChild(summary);

        if (action.reason && action.status === 'pending') {
            const reason = document.createElement('div');
            reason.className = 'approval-reason';
            reason.textContent = `Why: ${action.reason}.`;
            card.appendChild(reason);
        }

        if (action.result) {
            const result = document.createElement('div');
            result.className = 'approval-result';
            result.textContent = action.result.length > 1500
                ? action.result.slice(0, 1500) + '\n…'
                : action.result;
            card.appendChild(result);
        }

        const buttons = document.createElement('div');
        buttons.className = 'approval-actions';
        if (action.status === 'pending') {
            const deny = document.createElement('button');
            deny.textContent = 'Deny';
            deny.addEventListener('click', () => decideApproval(action.action_id, 'deny', card));
            const approve = document.createElement('button');
            approve.className = 'approve';
            approve.textContent = 'Approve';
            approve.addEventListener('click', () => decideApproval(action.action_id, 'approve', card));
            buttons.append(deny, approve);
        } else {
            const dismiss = document.createElement('button');
            dismiss.textContent = 'Dismiss';
            dismiss.addEventListener('click', () => {
                card.remove();
                approvalCards.delete(action.action_id);
            });
            buttons.append(dismiss);
        }
        card.appendChild(buttons);
    }

    async function decideApproval(actionId, decision, card) {
        card.querySelectorAll('button').forEach(b => { b.disabled = true; });
        try {
            const res = await fetch(`/api/actions/${actionId}/${decision}`, {
                method: 'POST',
                headers: authHeaders(),
            });
            const data = await res.json().catch(() => ({}));
            if (!res.ok) {
                triggerAlert(`G.I.D.E.O.N: ${data.detail || 'Could not update that request.'}`);
                // Already handled elsewhere (quick window, expiry): reflect it.
                loadPendingApprovals(currentSessionId, true);
                return;
            }
            renderApproval(data);
            triggerAlert(decision === 'approve'
                ? `G.I.D.E.O.N: Approved #${actionId}.`
                : `G.I.D.E.O.N: Denied #${actionId}.`);
            if (currentSessionId) loadHistory(currentSessionId);
        } catch (err) {
            console.error('Approval request failed:', err);
            card.querySelectorAll('button').forEach(b => { b.disabled = false; });
        }
    }

    async function loadPendingApprovals(sessionId, pruneResolved = false) {
        if (!sessionId) return;
        try {
            const res = await fetch(`/api/actions?session_id=${encodeURIComponent(sessionId)}`, {
                headers: authHeaders(),
            });
            if (!res.ok) return;
            const pending = await res.json();
            const pendingIds = new Set(pending.map(a => a.action_id));
            if (pruneResolved) {
                for (const [id, card] of approvalCards) {
                    if (!pendingIds.has(id) && !card.classList.contains('resolved')) {
                        card.remove();
                        approvalCards.delete(id);
                    }
                }
            }
            pending.forEach(renderApproval);
        } catch (err) {
            console.error('Failed to load pending approvals:', err);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  UI — REPLY FEEDBACK
    //  👍/👎 on the last reply, stored on that turn's trace — the
    //  correction signal the eval set and future training are built from.
    // ═══════════════════════════════════════════════════════════
    const replyFeedback = document.getElementById('reply-feedback');
    let lastTraceId = null;

    function showFeedback(traceId) {
        lastTraceId = traceId;
        if (!replyFeedback) return;
        replyFeedback.querySelectorAll('button').forEach(b => {
            b.disabled = false;
            b.classList.remove('chosen');
        });
        replyFeedback.classList.toggle('hidden', !traceId);
    }

    if (replyFeedback) {
        replyFeedback.querySelectorAll('button').forEach(button => {
            button.addEventListener('click', async () => {
                if (!lastTraceId) return;
                const rating = Number(button.dataset.rating);
                let note = null;
                if (rating < 0) {
                    note = prompt("What was wrong? (optional — helps G.I.D.E.O.N improve)");
                    if (note === null) return; // cancelled
                }
                try {
                    const res = await fetch(`/api/traces/${lastTraceId}/feedback`, {
                        method: 'POST',
                        headers: authHeaders(),
                        body: JSON.stringify({ rating, note: note || null }),
                    });
                    if (!res.ok) throw new Error(`HTTP ${res.status}`);
                    replyFeedback.querySelectorAll('button').forEach(b => { b.disabled = true; });
                    button.classList.add('chosen');
                    triggerAlert("G.I.D.E.O.N: Feedback recorded.");
                } catch (err) {
                    console.error('Feedback failed:', err);
                }
            });
        });
    }

    async function handleChatSend() {
        const query = chatInput.value.trim();
        if (!query) {
            triggerAlert("Please enter a query or diagnostic instruction.");
            return;
        }
        
        if (!currentSessionId) {
            triggerAlert("G.I.D.E.O.N: No active connection session.");
            return;
        }

        chatInput.value = '';
        triggerAlert("G.I.D.E.O.N: Analyzing query...");
        triggerReaction(3000); 
        
        const hudText = document.getElementById('hud-text');
        if (hudText) hudText.innerHTML = `<span style="opacity: 0.5; font-style: italic;">Analyzing query...</span>`;
        
        const logEl = document.getElementById('chat-history-log');
        if (logEl) {
            const bubble = document.createElement('div');
            bubble.className = 'chat-bubble chat-bubble-user';
            const content = document.createElement('div');
            content.className = 'chat-bubble-content';
            content.textContent = query;
            bubble.appendChild(content);
            const meta = document.createElement('div');
            meta.className = 'chat-bubble-meta';
            meta.textContent = `user | ${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;
            bubble.appendChild(meta);
            logEl.appendChild(bubble);
            
            const historyView = document.getElementById('history-view');
            if (historyView) historyView.scrollTop = historyView.scrollHeight;
        }

        stopAndClearAudio();
        showFeedback(null);
        let turnTraceId = null;
        try {
            const res = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: currentSessionId,
                    message: query
                })
            });
            
            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                const errMsg = errData.detail || "Unknown error";
                if (hudText) hudText.textContent = `Error: ${errMsg}`;
                triggerAlert("G.I.D.E.O.N: Core processing failure.");
                return;
            }
            
            // Set up UI for streaming
            if (hudText) hudText.textContent = '';
            const dot = document.querySelector('.hud-dot');
            if (dot) dot.classList.add('pulsing');
            
            const reader = res.body.getReader();
            const decoder = new TextDecoder('utf-8');
            let buffer = '';
            let fullText = '';
            
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();
                
                for (const line of lines) {
                    const stripped = line.trim();
                    if (!stripped || !stripped.startsWith('data:')) continue;
                    
                    try {
                        const parsed = JSON.parse(stripped.slice(5).trim());
                        if (parsed.text) {
                            fullText += parsed.text;
                            triggerReaction(500); // keep the model reactive during streaming
                            if (hudText) {
                                hudText.textContent = fullText;
                                const hudBody = hudText.parentElement;
                                if (hudBody) hudBody.scrollTop = hudBody.scrollHeight;
                            }
                        }
                        if (parsed.audio_url) {
                            enqueueAudio(parsed.audio_url);
                        }
                        if (parsed.trace_id) {
                            turnTraceId = parsed.trace_id;
                        }
                        if (parsed.approval) {
                            renderApproval(parsed.approval);
                            triggerAlert("G.I.D.E.O.N: An action is waiting for your approval.");
                        }
                        if (parsed.open_url) {
                            window.open(parsed.open_url, '_blank');
                        }
                        if (parsed.error) {
                            // The server could not raise an HTTP error once the
                            // stream had started, so it sends this instead.
                            console.error('G.I.D.E.O.N core error:', parsed.error);
                            if (hudText) hudText.textContent = fullText
                                ? `${fullText}\n\n[interrupted: ${parsed.error}]`
                                : `Error: ${parsed.error}`;
                            triggerAlert("G.I.D.E.O.N: Core processing failure.");
                        }
                    } catch (e) {
                        console.error('Failed to parse SSE line:', e);
                    }
                }
            }
            
            if (dot) dot.classList.remove('pulsing');
            showFeedback(turnTraceId);
            loadHistory(currentSessionId);
            triggerAlert("G.I.D.E.O.N: Core response received.");
            
        } catch (err) {
            console.error('Failed to communicate with G.I.D.E.O.N core:', err);
            if (hudText) hudText.textContent = "Error: Connection lost with G.I.D.E.O.N Core API. Check server console.";
            triggerAlert("G.I.D.E.O.N: Core link broken.");
        }
    }

    sendBtn.addEventListener('click', handleChatSend);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleChatSend();
    });

    // Session dropdown selection change (both header and sidebar)
    const selectEl = document.getElementById('session-select');
    const selectHeaderEl = document.getElementById('session-select-header');

    function handleSessionChange(sessionId) {
        currentSessionId = sessionId;
        if (selectEl) selectEl.value = sessionId;
        if (selectHeaderEl) selectHeaderEl.value = sessionId;
        loadHistory(sessionId);
        triggerAlert(`Resumed G.I.D.E.O.N session.`);
    }

    if (selectEl) {
        selectEl.addEventListener('change', (e) => {
            handleSessionChange(e.target.value);
        });
    }
    if (selectHeaderEl) {
        selectHeaderEl.addEventListener('change', (e) => {
            handleSessionChange(e.target.value);
        });
    }

    // New Session Button Click (both header and sidebar)
    const newSessionBtn = document.getElementById('new-session-btn');
    const newSessionBtnHeader = document.getElementById('new-session-btn-header');

    async function handleNewSession() {
        const name = prompt("Name the new session (leave blank and G.I.D.E.O.N will title it):");
        if (name === null) return; // User cancelled
        // Blank asks the server to title it after the first exchange.
        const sName = name.trim();

        try {
            const res = await fetch('/api/sessions', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name: sName })
            });
            const newSession = await res.json();
            triggerAlert(`G.I.D.E.O.N: Started new session: ${newSession.session_name}`);
            loadSessions(newSession.session_id);
        } catch (err) {
            console.error('Failed to create session:', err);
        }
    }

    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', handleNewSession);
    }
    if (newSessionBtnHeader) {
        newSessionBtnHeader.addEventListener('click', handleNewSession);
    }

    // Right Sidebar Navigation View Toggling
    const navItemContext = document.getElementById('nav-item-context');
    const navItemHistory = document.getElementById('nav-item-history');
    const metricsView    = document.getElementById('metrics-view');
    const historyView    = document.getElementById('history-view');
    const rightSidebarTitle = document.getElementById('right-sidebar-title');
    const rightSidebarSubtitle = document.getElementById('right-sidebar-subtitle');

    if (navItemContext && navItemHistory && metricsView && historyView) {
        navItemContext.addEventListener('click', () => {
            // Visual active state highlight toggling
            navItemContext.classList.add('text-neural-cyan');
            navItemContext.classList.remove('text-on-surface/30');
            navItemHistory.classList.add('text-on-surface/30');
            navItemHistory.classList.remove('text-neural-cyan');

            // Explicitly clear any legacy inline display styles
            metricsView.style.display = '';
            historyView.style.display = '';
            
            metricsView.classList.remove('hidden');
            historyView.classList.add('hidden');
            historyView.classList.remove('flex');
            
            if (rightSidebarTitle) rightSidebarTitle.textContent = "Metrics";
            if (rightSidebarSubtitle) rightSidebarSubtitle.textContent = "System Status Details";
            triggerAlert("G.I.D.E.O.N: Displaying system metrics.");
        });

        navItemHistory.addEventListener('click', () => {
            // Visual active state highlight toggling
            navItemHistory.classList.add('text-neural-cyan');
            navItemHistory.classList.remove('text-on-surface/30');
            navItemContext.classList.add('text-on-surface/30');
            navItemContext.classList.remove('text-neural-cyan');

            // Explicitly clear any legacy inline display styles
            metricsView.style.display = '';
            historyView.style.display = '';

            metricsView.classList.add('hidden');
            historyView.classList.remove('hidden');
            historyView.classList.add('flex');
            
            if (rightSidebarTitle) rightSidebarTitle.textContent = "History Log";
            if (rightSidebarSubtitle) rightSidebarSubtitle.textContent = "Active Chat Session Log";
            triggerAlert("G.I.D.E.O.N: Displaying conversation history.");
            if (currentSessionId) loadHistory(currentSessionId);
        });
    }

    micBtn.addEventListener('click', () => {
        triggerAlert("G.I.D.E.O.N: Voice initialization pending. Please connect model core.");
        triggerReaction(1500);
    });

    promptChips.forEach(chip => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.textContent.replace(/"/g, '');
            chatInput.focus();
            triggerAlert(`Selected prompt: ${chip.getAttribute('data-prompt')}`);
            triggerReaction(1000);
        });
    });

    // ═══════════════════════════════════════════════════════════
    //  UI — NAVIGATION LINKS
    // ═══════════════════════════════════════════════════════════
    const navs = [navDashboard, navNeural, navArchive];
    navs.forEach(nav => {
        if (nav) {
            nav.addEventListener('click', () => {
                navs.forEach(n => n.classList.remove('active'));
                nav.classList.add('active');
                triggerAlert(`Loading: ${nav.textContent} panel...`);
            });
        }
    });

    // Persistent EventSource listener for proactive system notifications (reminders & terminal events)
    function initProactiveEvents() {
        try {
            const eventSource = new EventSource('/api/events');
            eventSource.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    if (data.type === 'reminder' || data.type === 'reminder_alert') {
                        const alertMsg = `🔔 REMINDER: ${data.text}`;
                        triggerAlert(alertMsg);
                        triggerReaction(2000);
                        const hud = document.getElementById('hud-text');
                        if (hud) hud.textContent = alertMsg;
                        if (data.audio_url) enqueueAudio(data.audio_url);
                    } else if (data.type === 'reminder_digest') {
                        const digestMsg = `🔔 ${data.message}: ` +
                            (data.items || []).map(i => i.text).join(' · ');
                        triggerAlert(`🔔 ${data.message}`);
                        triggerReaction(2000);
                        const hud = document.getElementById('hud-text');
                        if (hud) hud.textContent = digestMsg;
                        if (data.audio_url) enqueueAudio(data.audio_url);
                    } else if (data.type === 'terminal') {
                        toggleTerminalDrawer(true);
                        appendTerminalOutput(data.command, data.output, data.returncode);
                    } else if (data.type === 'action_resolved') {
                        // Resolved here or in the quick window: keep cards in step.
                        if (approvalCards.has(data.action_id)) renderApproval(data);
                    } else if (data.type === 'power_changed') {
                        onBattery = !!data.on_battery;
                        const mode = { cpu: 'model moved to CPU', off: 'model paused', gpu: 'model unchanged' };
                        triggerAlert(data.on_battery
                            ? `G.I.D.E.O.N: On battery — ${mode[data.battery_mode] || 'power saving'}.`
                            : 'G.I.D.E.O.N: On mains power — GPU available.');
                    } else if (data.type === 'session_renamed') {
                        loadSessions(currentSessionId);
                    }
                } catch (e) {
                    console.error("Failed to parse proactive SSE event:", e);
                }
            };
            eventSource.onerror = (err) => {
                console.warn("Proactive SSE stream lost, retrying in 5s...", err);
                eventSource.close();
                setTimeout(initProactiveEvents, 5000);
            };
        } catch (e) {
            console.error("Could not initialize EventSource /api/events:", e);
        }
    }

    // ═══════════════════════════════════════════════════════════
    //  TERMINAL HUD DRAWER & COMMAND LOGGING
    // ═══════════════════════════════════════════════════════════
    const btnTerminalToggle = document.getElementById('btn-terminal-toggle');
    const terminalDrawer = document.getElementById('terminal-drawer');
    const terminalCloseBtn = document.getElementById('terminal-close-btn');
    const terminalClearBtn = document.getElementById('terminal-clear-btn');
    const terminalOutput = document.getElementById('terminal-output');
    const terminalInputForm = document.getElementById('terminal-input-form');
    const terminalInput = document.getElementById('terminal-input');

    function toggleTerminalDrawer(forceState) {
        if (!terminalDrawer) return;
        const isOpen = terminalDrawer.classList.contains('open');
        const shouldOpen = forceState !== undefined ? forceState : !isOpen;
        if (shouldOpen) {
            terminalDrawer.classList.add('open');
            if (terminalInput) terminalInput.focus();
        } else {
            terminalDrawer.classList.remove('open');
        }
    }

    function appendTerminalOutput(command, output, returncode) {
        if (!terminalOutput) return;
        const entry = document.createElement('div');
        entry.className = 'space-y-1 pb-2 border-b border-neural-cyan/10 text-[11px]';

        const promptLine = document.createElement('div');
        promptLine.className = 'flex items-center gap-2 font-bold text-neural-cyan';
        const timeSpan = document.createElement('span');
        timeSpan.className = 'text-on-surface/40';
        timeSpan.textContent = `[${new Date().toLocaleTimeString()}]`;
        const cmdSpan = document.createElement('span');
        cmdSpan.textContent = `$ ${command}`;
        promptLine.appendChild(timeSpan);
        promptLine.appendChild(cmdSpan);

        const outLine = document.createElement('pre');
        const codeColor = (returncode === 0 || returncode === undefined) ? 'text-on-surface/90' : 'text-system-red';
        outLine.className = `whitespace-pre-wrap ${codeColor} pl-2 border-l-2 ${returncode === 0 ? 'border-neural-cyan/40' : 'border-system-red/50'} mt-1 font-data-mono`;
        outLine.textContent = output || '(Command completed with no output)';

        entry.appendChild(promptLine);
        entry.appendChild(outLine);
        terminalOutput.appendChild(entry);
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
    }

    const navItemTerminal = document.getElementById('nav-item-terminal');
    if (navItemTerminal) {
        navItemTerminal.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            toggleTerminalDrawer();
        });
    }

    if (btnTerminalToggle) {
        btnTerminalToggle.addEventListener('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            toggleTerminalDrawer();
        });
    }

    // Global keyboard shortcut: Ctrl+` (Control + Backtick)
    window.addEventListener('keydown', (e) => {
        if (e.ctrlKey && (e.key === '`' || e.code === 'Backquote')) {
            e.preventDefault();
            toggleTerminalDrawer();
        }
    });
    if (terminalCloseBtn) {
        terminalCloseBtn.addEventListener('click', () => toggleTerminalDrawer(false));
    }
    if (terminalClearBtn) {
        terminalClearBtn.addEventListener('click', () => {
            if (terminalOutput) {
                terminalOutput.innerHTML = '<div class="text-on-surface/40 italic">// G.I.D.E.O.N Live Execution Stream Initialized.</div>';
            }
        });
    }

    if (terminalInputForm) {
        terminalInputForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            if (!terminalInput) return;
            const cmd = terminalInput.value.trim();
            if (!cmd) return;
            terminalInput.value = '';

            try {
                const res = await fetch('/api/terminal/execute', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        // Injected into the page by the server's index route.
                        'X-Gideon-Token': window.GIDEON_API_TOKEN || '',
                    },
                    body: JSON.stringify({ command: cmd }),
                });
                if (res.status === 401) {
                    appendTerminalOutput(cmd, 'Unauthorized: reload the page to refresh the API token.', 1);
                    return;
                }
                const data = await res.json();
                appendTerminalOutput(cmd, data.output, data.status === 'ok' ? 0 : 1);
            } catch (err) {
                appendTerminalOutput(cmd, `Execution failed: ${err.message}`, 1);
            }
        });
    }

    // Initialize Sessions, Chat Connection, and Proactive Events on load
    loadSessions();
    initProactiveEvents();
});
