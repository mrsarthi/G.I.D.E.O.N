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

void main() {
    vNormal   = normalize(normalMatrix * normal);
    vPos      = position;
    vWorldPos = (modelMatrix * vec4(position, 1.0)).xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
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

    // ═══════════════════════════════════════════════════════════
    //  THREE.JS SCENE SETUP
    // ═══════════════════════════════════════════════════════════
    const container = document.getElementById('canvas-container');
    const W = container.clientWidth  || 600;
    const H = container.clientHeight || 600;

    const scene  = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, W / H, 0.1, 1000);
    camera.position.set(0, 0.1, 4.5);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
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
            uIntensity: { value: 0.88 }
        },
        vertexShader:   HOLO_VERT,
        fragmentShader: HOLO_FRAG,
        transparent: true,
        side:        THREE.DoubleSide,
        depthWrite:  false,
        blending:    THREE.AdditiveBlending
    });

    // Wireframe material for overlay
    const wireMat = new THREE.MeshBasicMaterial({
        color: 0x5ffbd6, wireframe: true, transparent: true,
        opacity: 0.07, blending: THREE.AdditiveBlending
    });

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

    function animate() {
        requestAnimationFrame(animate);
        const t = threeClock.getElapsedTime();

        // Shader uniforms
        holoMat.uniforms.uTime.value = t;

        // Reactive intensity decay
        reactiveBoost *= 0.97;
        holoMat.uniforms.uIntensity.value = 0.88 + reactiveBoost * 0.4;

        // ── Head breathing / float ──
        headGroup.position.y = 0.45 + Math.sin(t * 1.2) * 0.025;

        // ── Smooth head rotation (gaze tracking) ──
        headGroup.rotation.y += (targetRotY - headGroup.rotation.y) * 0.08;
        headGroup.rotation.x += (targetRotX - headGroup.rotation.x) * 0.08;

        // ── Wireframe pulse ──
        wireMat.opacity = 0.06 + Math.sin(t * 1.8) * 0.02;

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
    function simulateMetrics() {
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
        ramVal.innerHTML = `${ram} GB <span>/ 16GB</span>`;
    }
    setInterval(simulateMetrics, 3000);
    simulateMetrics();

    // ═══════════════════════════════════════════════════════════
    //  UI — MOUSE PARALLAX & ROTATION
    // ═══════════════════════════════════════════════════════════
    document.addEventListener('mousemove', (e) => {
        const x = (e.clientX - window.innerWidth  / 2) / 100;
        const y = (e.clientY - window.innerHeight / 2) / 100;

        leftSidebar.style.transform       = `translate(${x * 1.5}px, calc(-50% + ${y * 1.5}px))`;
        rightSidebar.style.transform      = `translate(${x * -1.5}px, calc(-50% + ${y * -1.5}px))`;
        hologramContainer.style.transform = `translate(${x * 3}px, ${y * 3}px)`;

        // Gaze tracking rotation target
        const mx = (e.clientX / window.innerWidth) * 2 - 1;
        const my = -(e.clientY / window.innerHeight) * 2 + 1;
        targetRotY = mx * 0.45;  // Look left/right
        targetRotX = -my * 0.25; // Look up/down
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
            if (!selectEl) return;

            selectEl.innerHTML = '';
            
            if (sessions.length === 0) {
                const newRes = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name: 'Default G.I.D.E.O.N Session' })
                });
                const defaultSession = await newRes.json();
                currentSessionId = defaultSession.session_id;
                
                const opt = document.createElement('option');
                opt.value = defaultSession.session_id;
                opt.textContent = defaultSession.session_name;
                selectEl.appendChild(opt);
            } else {
                sessions.forEach(s => {
                    const opt = document.createElement('option');
                    opt.value = s.session_id;
                    opt.textContent = s.session_name;
                    selectEl.appendChild(opt);
                });
                
                if (selectAfterLoadId) {
                    currentSessionId = selectAfterLoadId;
                } else if (!currentSessionId) {
                    currentSessionId = sessions[0].session_id;
                }
                selectEl.value = currentSessionId;
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
                    } catch (e) {
                        console.error('Failed to parse SSE line:', e);
                    }
                }
            }
            
            if (dot) dot.classList.remove('pulsing');
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

    // Session dropdown selection change
    const selectEl = document.getElementById('session-select');
    if (selectEl) {
        selectEl.addEventListener('change', (e) => {
            currentSessionId = e.target.value;
            loadHistory(currentSessionId);
            triggerAlert(`Resumed G.I.D.E.O.N session.`);
        });
    }

    // New Session Button Click
    const newSessionBtn = document.getElementById('new-session-btn');
    if (newSessionBtn) {
        newSessionBtn.addEventListener('click', async () => {
            const name = prompt("Enter a name for the new session:");
            const sName = name ? name.trim() : `Session ${new Date().toLocaleDateString()} ${new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'})}`;
            
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
        });
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
            navItemContext.classList.add('active');
            navItemHistory.classList.remove('active');
            metricsView.style.display = 'flex';
            historyView.style.display = 'none';
            if (rightSidebarTitle) rightSidebarTitle.textContent = "Metrics";
            if (rightSidebarSubtitle) rightSidebarSubtitle.textContent = "System Status Details";
            triggerAlert("G.I.D.E.O.N: Displaying system metrics.");
        });

        navItemHistory.addEventListener('click', () => {
            navItemHistory.classList.add('active');
            navItemContext.classList.remove('active');
            metricsView.style.display = 'none';
            historyView.style.display = 'flex';
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

    // Initialize Sessions and Chat Connection on load
    loadSessions();
});
