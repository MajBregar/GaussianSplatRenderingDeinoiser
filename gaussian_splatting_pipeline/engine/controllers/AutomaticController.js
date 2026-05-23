import { quat, vec3 } from 'glm';
import { Transform } from '../core/Transform.js';
export class AutomaticController {

    constructor(node, domElement, {
        rotationRate = [0.01, 0, 0],
        distanceRate = 0,
        target = [0, 0, 0],
        angles = [0, 0, 0],
        distance = 2,
    } = {}) {
        this.node = node;
        this.domElement = domElement;
        this.rotationRate = rotationRate;
        this.distanceRate = distanceRate;
        this.rotating = false;
        this.target = vec3.clone(target);
        this.distance = distance;

        this.rotation = quat.create();
        quat.rotateX(this.rotation, this.rotation, angles[0] * Math.PI / 180);
        quat.rotateY(this.rotation, this.rotation, angles[1] * Math.PI / 180);
        quat.rotateZ(this.rotation, this.rotation, angles[2] * Math.PI / 180);

        let transform = node.getComponentOfType(Transform);
        if (!transform) {
            transform = new Transform();
            node.addComponent(transform);
        }

        this.initHandlers();
    }

    initHandlers() {
        this.keydownHandler = this.keydownHandler.bind(this);
        document.addEventListener('keydown', this.keydownHandler);
    }

    keydownHandler(e) {
        if (e.code === 'Space') {
            e.preventDefault();
            this.rotating = !this.rotating;
        }
    }

    togglePause() {
        this.rotating = !this.rotating;
    }

    pause() {
        this.rotating = false;
    }

    resume() {
        this.rotating = true;
    }


    update() {
        const transform = this.node.getComponentOfType(Transform);
        if (!transform) return;

        if (this.rotating) {
            const [rx, ry, rz] = this.rotationRate;
            const rot = quat.create();
            quat.rotateX(rot, rot, rx);
            quat.rotateY(rot, rot, ry);
            quat.rotateZ(rot, rot, rz);
            quat.multiply(this.rotation, rot, this.rotation);
            quat.normalize(this.rotation, this.rotation);

            this.distance += this.distanceRate;
        }

        const offset = vec3.transformQuat(vec3.create(), [0, 0, this.distance], this.rotation);
        vec3.add(transform.translation, this.target, offset);
        quat.copy(transform.rotation, this.rotation);
    }

}