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
