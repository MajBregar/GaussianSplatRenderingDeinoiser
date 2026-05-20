export class PerformanceTracker {
    constructor(windowSize = 60) {
        this.windowSize = windowSize;
        this.sections = {};
    }

    begin(name) {
        if (!this.sections[name]) {
            this.sections[name] = { samples: [], avg: 0, _t0: 0 };
        }
        this.sections[name]._t0 = performance.now();
    }

    end(name) {
        const s = this.sections[name];
        if (!s) return;
        const elapsed = performance.now() - s._t0;
        s.samples.push(elapsed);
        if (s.samples.length > this.windowSize) s.samples.shift();
        s.avg = s.samples.reduce((a, b) => a + b, 0) / s.samples.length;
    }

    get(name) {
        return this.sections[name]?.avg ?? 0;
    }

    summary() {
        return Object.fromEntries(
            Object.entries(this.sections).map(([k, v]) => [k, v.avg.toFixed(2) + 'ms'])
        );
    }
}