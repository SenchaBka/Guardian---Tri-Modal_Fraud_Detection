module.exports = {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        bg: { 0:'#07080b', 1:'#0d0f14', 2:'#13161d', 3:'#1a1e28', 4:'#222736' },
        border: { subtle:'rgba(255,255,255,0.06)', DEFAULT:'rgba(255,255,255,0.1)', strong:'rgba(255,255,255,0.18)' },
        teal:   { DEFAULT:'#00e5c3', dim:'rgba(0,229,195,0.1)', border:'rgba(0,229,195,0.2)', dark:'#00b89c' },
        amber:  { DEFAULT:'#f5a623', dim:'rgba(245,166,35,0.1)', border:'rgba(245,166,35,0.2)' },
        danger: { DEFAULT:'#ff4757', dim:'rgba(255,71,87,0.1)', border:'rgba(255,71,87,0.2)' },
        sky:    { DEFAULT:'#4a9eff', dim:'rgba(74,158,255,0.08)' },
        ink:    { DEFAULT:'#dde1ec', muted:'#6b7494', dim:'#3a3f55' },
      },
      fontFamily: {
        display: ['"Bricolage Grotesque"', 'sans-serif'],
        mono:    ['"DM Mono"', 'monospace'],
        sans:    ['"Bricolage Grotesque"', 'sans-serif'],
      },
      animation: {
        'pulse-dot': 'pulse-dot 2s ease-in-out infinite',
        'spin-slow': 'spin 0.7s linear infinite',
        'fade-in':   'fadeIn 0.2s ease-out',
        'slide-up':  'slideUp 0.25s ease-out',
      },
      keyframes: {
        'pulse-dot': { '0%,100%':{ opacity:'1', boxShadow:'0 0 0 0 rgba(0,229,195,0.4)' }, '50%':{ opacity:'0.8', boxShadow:'0 0 0 5px rgba(0,229,195,0)' } },
        fadeIn:  { from:{ opacity:'0' }, to:{ opacity:'1' } },
        slideUp: { from:{ opacity:'0', transform:'translateY(8px)' }, to:{ opacity:'1', transform:'translateY(0)' } },
      },
    },
  },
  plugins: [],
}