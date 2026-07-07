interface Props {
  name: IconName
  size?: number
  className?: string
  filled?: boolean
}

export type IconName =
  | 'logo'
  | 'sun'
  | 'moon'
  | 'camera'
  | 'close'
  | 'star'
  | 'power'
  | 'faces'
  | 'trash'
  | 'check'
  | 'edit'
  | 'blend'
  | 'swap'
  | 'wand'
  | 'sparkles'
  | 'shield'
  | 'github'
  | 'upload'
  | 'download'
  | 'photo'
  | 'heart'
  | 'share'

// Minimal inline SVG set — stroke uses currentColor so icons inherit text color.
const PATHS: Record<IconName, JSX.Element> = {
  logo: (
    <>
      <circle cx="9" cy="12" r="6" />
      <circle cx="15" cy="12" r="6" />
    </>
  ),
  sun: (
    <>
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
    </>
  ),
  moon: <path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z" />,
  camera: (
    <>
      <path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L19 6h0a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
      <circle cx="12" cy="12.5" r="3.5" />
    </>
  ),
  close: <path d="M6 6l12 12M18 6L6 18" />,
  star: <path d="M12 3l2.6 5.3 5.9.9-4.2 4.1 1 5.8L12 16.9 6.7 19.1l1-5.8L3.5 9.2l5.9-.9z" />,
  power: (
    <>
      <path d="M12 3v9" />
      <path d="M6.6 6.6a8 8 0 1 0 10.8 0" />
    </>
  ),
  faces: (
    <>
      <circle cx="9" cy="10" r="6" />
      <circle cx="16" cy="14" r="6" />
    </>
  ),
  trash: <path d="M4 7h16M9 7V5a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2M6 7l1 13a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-13" />,
  check: <path d="M5 12.5l4 4 10-10" />,
  edit: (
    <>
      <path d="M4 20h4L18.5 9.5a2 2 0 0 0-2.83-2.83L5 17.2z" />
      <path d="M14.5 7.5l2.8 2.8" />
    </>
  ),
  blend: (
    <>
      <circle cx="7" cy="12" r="4" />
      <circle cx="17" cy="12" r="4" />
      <path d="M11 8.5c1.7 1.9 1.7 5.1 0 7M13 8.5c-1.7 1.9-1.7 5.1 0 7" />
    </>
  ),
  swap: (
    <>
      <path d="M7 16V4M7 4L3.5 7.5M7 4l3.5 3.5" />
      <path d="M17 8v12M17 20l3.5-3.5M17 20l-3.5-3.5" />
    </>
  ),
  wand: (
    <>
      <path d="M3 21L14.5 9.5" />
      <path d="M16.5 2.5v3M16.5 10.5v3M12 8h3M18 8h3M14.2 5.7l-1.4-1.4M20.2 11.7l-1.4-1.4M20.2 4.3l-1.4 1.4" />
    </>
  ),
  sparkles: (
    <>
      <path d="M12 3l1.7 4.6L18.3 9.3l-4.6 1.7L12 15.6l-1.7-4.6L5.7 9.3l4.6-1.7z" />
      <path d="M19 14.5l.8 2.2 2.2.8-2.2.8-.8 2.2-.8-2.2-2.2-.8 2.2-.8z" />
    </>
  ),
  shield: <path d="M12 3l7 2.8v5.4c0 4.3-2.9 7.7-7 9.8-4.1-2.1-7-5.5-7-9.8V5.8z" />,
  github: (
    <path
      fill="currentColor"
      stroke="none"
      d="M12 2C6.48 2 2 6.58 2 12.25c0 4.53 2.87 8.37 6.84 9.73.5.09.68-.22.68-.49 0-.24-.01-.88-.01-1.73-2.78.62-3.37-1.37-3.37-1.37-.45-1.18-1.11-1.5-1.11-1.5-.91-.63.07-.62.07-.62 1 .07 1.53 1.06 1.53 1.06.89 1.57 2.34 1.11 2.91.85.09-.66.35-1.11.63-1.37-2.22-.26-4.56-1.14-4.56-5.07 0-1.12.39-2.03 1.03-2.75-.1-.26-.45-1.3.1-2.7 0 0 .84-.28 2.75 1.05a9.36 9.36 0 0 1 5 0c1.91-1.33 2.75-1.05 2.75-1.05.55 1.4.2 2.44.1 2.7.64.72 1.03 1.63 1.03 2.75 0 3.94-2.34 4.8-4.57 5.06.36.32.68.94.68 1.9 0 1.37-.01 2.47-.01 2.81 0 .27.18.59.69.49A10.25 10.25 0 0 0 22 12.25C22 6.58 17.52 2 12 2z"
    />
  ),
  upload: <path d="M12 16V4M12 4L7.5 8.5M12 4l4.5 4.5M4 20h16" />,
  download: <path d="M12 4v12M12 16l-4.5-4.5M12 16l4.5-4.5M4 20h16" />,
  photo: (
    <>
      <rect x="3" y="5" width="18" height="14" rx="3" />
      <circle cx="9" cy="10" r="1.6" />
      <path d="M4 17l4.5-4.5 3 3L16 11l4 4" />
    </>
  ),
  heart: (
    <path d="M12 20s-7-4.4-9.3-8.3C1.2 8.9 2.5 5.5 5.7 5.5c1.9 0 3.2 1.1 4.1 2.4l.9 1.3.9-1.3c.9-1.3 2.2-2.4 4.1-2.4 3.2 0 4.5 3.4 3 6.2C19 15.6 12 20 12 20z" />
  ),
  share: (
    <>
      <circle cx="6" cy="12" r="2.5" />
      <circle cx="17" cy="6" r="2.5" />
      <circle cx="17" cy="18" r="2.5" />
      <path d="M8.2 10.8l6.6-3.6M8.2 13.2l6.6 3.6" />
    </>
  ),
}

export function Icon({ name, size = 18, className = '', filled = false }: Props) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={filled ? 'currentColor' : 'none'}
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      {PATHS[name]}
    </svg>
  )
}
